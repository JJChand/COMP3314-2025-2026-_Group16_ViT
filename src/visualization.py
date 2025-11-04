import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import os
import json
from train import TrainingHistory

class ViTVisualizer:
    """ViT模型可视化类 - 适配新的TrainingHistory格式"""
    
    def __init__(self, checkpoint_dir='./checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = None
    
    def load_history_from_checkpoint(self, checkpoint_path=None):
        """从模型检查点加载训练历史"""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        print(f"从检查点加载历史: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 从检查点中提取训练历史
        self.history = checkpoint.get('training_history', {})
        self.model_args = checkpoint.get('args', None)
        self.best_acc = checkpoint.get('best_acc', 0)
        
        if not self.history:
            print("警告: 检查点中未找到训练历史")
        
        return self.history
    
    def load_history_from_json(self, json_path=None):
        """从JSON文件加载训练历史"""
        if json_path is None:
            json_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON历史文件不存在: {json_path}")
        
        print(f"从JSON文件加载历史: {json_path}")
        
        # 创建TrainingHistory对象并加载
        self.history_obj = TrainingHistory()
        self.history_obj.load(json_path)
        
        # 转换为字典格式以保持兼容性
        self.history = {
            'epochs': self.history_obj.epochs,
            'train_losses': self.history_obj.train_losses,
            'train_accs': self.history_obj.train_accs,
            'val_losses': self.history_obj.val_losses,
            'val_accs': self.history_obj.val_accs,
            'learning_rates': self.history_obj.learning_rates
        }
        
        return self.history
    
    def plot_training_progress(self, save_path=None):
        """绘制训练进度图"""
        if self.history is None:
            print("请先加载训练历史")
            return
        
        if not self.history.get('epochs'):
            print("训练历史数据为空")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.history['epochs']
        
        # 1. 损失曲线
        ax1.plot(epochs, self.history['train_losses'], 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, self.history['val_losses'], 'r-', label='验证损失', linewidth=2)
        ax1.set_xlabel('训练轮次')
        ax1.set_ylabel('损失值')
        ax1.set_title('训练和验证损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率曲线
        ax2.plot(epochs, self.history['train_accs'], 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, self.history['val_accs'], 'r-', label='验证准确率', linewidth=2)
        ax2.set_xlabel('训练轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.set_title('训练和验证准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 学习率变化
        ax3.plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
        ax3.set_xlabel('训练轮次')
        ax3.set_ylabel('学习率')
        ax3.set_title('学习率变化')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. 过拟合分析
        overfit_gap = [train - val for train, val in 
                      zip(self.history['train_accs'], self.history['val_accs'])]
        ax4.plot(epochs, overfit_gap, 'orange', linewidth=2)
        ax4.set_xlabel('训练轮次')
        ax4.set_ylabel('过拟合差距 (%)')
        ax4.set_title('训练-验证差距 (过拟合指标)')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=np.mean(overfit_gap), color='red', linestyle='--', 
                   label=f'平均差距: {np.mean(overfit_gap):.2f}%')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练进度图已保存: {save_path}")
        
        plt.show()
        return fig
    
    def create_performance_table(self):
        """创建性能指标表格"""
        if self.history is None or not self.history.get('epochs'):
            print("没有可用的训练历史数据")
            return None
        
        epochs = self.history['epochs']
        train_accs = self.history['train_accs']
        val_accs = self.history['val_accs']
        train_losses = self.history['train_losses']
        val_losses = self.history['val_losses']
        
        # 选择关键epoch点
        if len(epochs) >= 5:
            key_indices = [0, len(epochs)//4, len(epochs)//2, len(epochs)*3//4, -1]
        else:
            key_indices = range(len(epochs))
        
        table_data = []
        for idx in key_indices:
            if idx < len(epochs):
                table_data.append({
                    'Epoch': epochs[idx],
                    'Train Acc (%)': f"{train_accs[idx]:.2f}",
                    'Val Acc (%)': f"{val_accs[idx]:.2f}",
                    'Train Loss': f"{train_losses[idx]:.4f}",
                    'Val Loss': f"{val_losses[idx]:.4f}",
                    'Overfitting Gap': f"{train_accs[idx] - val_accs[idx]:.2f}"
                })
        
        df = pd.DataFrame(table_data)
        return df
    
    def ablation_analysis(self):
        """消融实验分析"""
        if self.history is None:
            print("请先加载训练历史")
            return
        
        if not self.history.get('epochs'):
            print("训练历史数据为空")
            return
        
        train_accs = self.history['train_accs']
        val_accs = self.history['val_accs']
        
        # 分析训练动态
        final_train_acc = train_accs[-1]
        final_val_acc = val_accs[-1]
        best_val_acc = max(val_accs)
        best_epoch = val_accs.index(best_val_acc) + 1
        
        print("=" * 60)
        print("消融实验分析报告")
        print("=" * 60)
        
        print(f"\n1. 模型性能分析:")
        print(f"   • 最终训练准确率: {final_train_acc:.2f}%")
        print(f"   • 最终验证准确率: {final_val_acc:.2f}%")
        print(f"   • 最佳验证准确率: {best_val_acc:.2f}% (第{best_epoch}轮)")
        print(f"   • 过拟合程度: {final_train_acc - final_val_acc:.2f}%")

        return {
            'final_train_acc': final_train_acc,
            'final_val_acc': final_val_acc,
            'best_val_acc': best_val_acc,
            'overfitting_gap': final_train_acc - final_val_acc
        }
