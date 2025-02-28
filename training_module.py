import torch
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch_xla as xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp



def train_model(model, train_loader, val_loader, loss_func, optimizer,
                epochs=10, schedulers=None, clip_grad=False, clip_value=1.0,
                use_tta=False, tta_transforms=None):
    """
    Train a model with optional gradient clipping and TTA evaluation.
    
    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        loss_func: Loss function.
        optimizer: Optimizer.
        epochs: Number of training epochs.
        schedulers: List of learning rate schedulers.
        clip_grad (bool): If True, perform gradient clipping.
        clip_value (float): Maximum gradient norm.
        use_tta (bool): If True, evaluate validation using TTA.
        tta_transforms: List of TTA transforms to use if use_tta is True.
    
    Returns:
        The trained model.
    """
    def plot_losses(train_losses, val_losses):
        """Function to plot metrics."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training and Validation Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def tta_predict(model, image, tta_transforms, device):
        """
        Perform test-time augmentation for a single image.
        
        Args:
            model: Trained PyTorch model in eval mode.
            image: A PIL Image or tensor representing the input.
            tta_transforms: A list of torchvision transforms (as lambda functions) to apply.
            device: Device to run inference on.
        
        Returns:
            Averaged prediction tensor.
        """
        model.eval()
        preds = []
        base_transform = transforms.ToTensor()
        # Convert to tensor if needed
        if not isinstance(image, torch.Tensor):
            image_tensor = base_transform(image)
        else:
            image_tensor = image
        with torch.inference_mode():
            for t in tta_transforms:
                augmented = t(image_tensor)
                # Make sure we have a batch dimension
                if len(augmented.shape) == 3:
                    augmented = augmented.unsqueeze(0)
                augmented = augmented.to(device)
                pred = model(augmented)
                preds.append(pred)
                xm.mark_step() 
        avg_pred = torch.mean(torch.stack(preds), dim=0)
        return avg_pred

    def evaluate_model(model, data_loader, loss_fn, tta=False, tta_transforms=None):
        """
        Evaluate the model on a data loader with optional TTA.
        
        Args:
            model: The trained model.
            data_loader: DataLoader for evaluation.
            loss_fn: Loss function.
            tta (bool): If True, apply TTA evaluation.
            tta_transforms: List of TTA transforms (if tta is True).
        
        Returns:
            (Average loss, Accuracy)
        """
        model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.inference_mode():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)
                if tta and tta_transforms is not None:
                    # Process each image individually with TTA
                    batch_preds = []
                    for i in range(images.size(0)):
                        img = images[i]
                        pred = tta_predict(model, img, tta_transforms, device)
                        batch_preds.append(pred)
                    outputs = torch.cat(batch_preds, dim=0)
                else:
                    outputs = model(images)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total_samples += images.size(0)
    
                xm.mark_step() 
        return total_loss / total_samples, correct / total_samples
    device = xm.xla_device()
                  
    best_acc = 0.0
    model = model.to(device)
    epochs_no_improve = 0
    early_stop = False
    
    # Lists to store loss values
    train_losses , val_losses = [] , []
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        
        if early_stop:
            print(f"Early stop activated at epoch {epoch+1}")
            break
        
        for images, labels in train_loader:
            with xm.step():
                images , labels = images.to(device), labels.to(device)
             
            
                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_func(outputs, labels)
                loss.backward()

                # Apply gradient clipping if enabled
                if clip_grad:
                    # This clips gradients of all parameters so that their norm does not exceed clip_value
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)
                
                #optimizer.step()
                xm.optimizer_step(optimizer)
            
                running_loss += loss.item() * images.size(0)

        
        # Calculate average training loss for the epoch
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, loss_func, tta=use_tta, tta_transforms=tta_transforms)
        val_losses.append(val_loss)
        
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

        if schedulers is not None:
            for scheduler in schedulers:
                scheduler.step(val_acc)
 

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{MODEL_NAME}_{experiment}.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                early_stop = True
                
    # Plot losses after training
    plot_losses(train_losses, val_losses)
    print(f"Best val acc: {best_acc:.4f}")
    return model
