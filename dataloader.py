import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class APSEDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): 项目根目录 (例如 E:/pythonProject)
            transform (callable, optional): 图像预处理 (Resize, Normalize等)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_list = []

        # 定义8个情感类别 (对应文件夹名称)
        self.emotions = [
            'Amusement', 'Anger', 'Awe', 'Contentment',
            'Disgust', 'Excitement', 'Fear', 'Sadness'
        ]

        # 建立标签索引 (Amusement -> 0, Anger -> 1 ...)
        self.label_map = {label: idx for idx, label in enumerate(self.emotions)}

        # 1. 加载所有描述文件 (句子_*.txt)
        # 建立映射: 纯文件名ID -> 描述文本
        print("Step 1: Loading captions from text files...")
        self.desc_map = self._load_all_descriptions()
        print(f" -> Loaded {len(self.desc_map)} captions.")

        # 2. 扫描文件夹，构建数据集
        # 逻辑: 遍历 root_dir/情感名/ 下的所有图片 -> 匹配描述 -> 加入列表
        print("Step 2: Scanning images and matching with captions...")
        self._build_dataset()

        print(f"Dataset initialization complete.")
        print(f"Total valid pairs (Image + Caption): {len(self.data_list)}")

    def _load_all_descriptions(self):
        """
        扫描根目录下所有的 '句子_*.txt' 文件并加载到内存
        """
        desc_map = {}
        # 匹配 句子_1.txt ... 句子_8.txt 以及可能的 句子.txt
        pattern = os.path.join(self.root_dir, "句子*.txt")
        files = glob.glob(pattern)

        if not files:
            print(f"Warning: No description files found matching {pattern}")
            return desc_map

        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 格式: filename.jpg [SEP] description
                    if '[SEP]' in line:
                        parts = line.split('[SEP]')
                        filename_with_ext = parts[0].strip()
                        description = parts[1].strip()

                        # 提取核心ID (去除后缀 .jpg/.json 等，处理 _z 后缀)
                        # 例如: "14017061191_..._z.jpg" -> "14017061191_..._z"
                        file_id = os.path.splitext(filename_with_ext)[0]
                        desc_map[file_id] = description
        return desc_map

    def _build_dataset(self):
        """
        遍历8个情感文件夹，匹配图片和描述
        """
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

        for label in self.emotions:
            # 假设图片路径: E:/pythonProject/Amusement/
            class_dir = os.path.join(self.root_dir, label)

            if not os.path.isdir(class_dir):
                print(f"Warning: Directory not found: {class_dir}")
                continue

            # 遍历文件夹内的文件
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(valid_exts):
                    # 提取文件ID
                    file_id = os.path.splitext(filename)[0]

                    # 只有当图片有对应的描述时，才加入数据集
                    if file_id in self.desc_map:
                        img_path = os.path.join(class_dir, filename)

                        self.data_list.append({
                            'img_path': img_path,
                            'caption': self.desc_map[file_id],  # 对应 RoBERTa 输入
                            'label': label,
                            'label_idx': self.label_map[label],
                            'file_id': file_id
                        })

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]

        # 1. 加载图像 (Visual Feature Input)
        try:
            image = Image.open(item['img_path']).convert('RGB')
        except Exception as e:
            print(f"Error loading {item['img_path']}: {e}")
            # 容错：生成纯黑图，防止训练中断
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        # 2. 返回数据
        # caption 返回原始字符串，将在 main.py 中通过 Tokenizer 统一处理成 180 长度
        return {
            'image': image,  # Tensor [3, 224, 224]
            'caption': item['caption'],  # String
            'label': item['label'],  # Str (e.g., 'Amusement')
            'label_idx': item['label_idx'],  # Int (e.g., 0)
            'id': item['file_id']
        }


# --- 测试配置代码 (验证维度是否符合您的要求) ---
if __name__ == "__main__":
    # 1. 设置 Visual Transform (匹配 CLIP-B/32 或 ViT 的归一化要求)
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711))
    ])

    # 请确保路径正确
    root_path = r"E:\pythonProject"

    print("Initializing Dataset...")
    dataset = APSEDataset(root_dir=root_path, transform=transform)

    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print("\n--- Success! Testing Batch ---")

        # 模拟 main.py 中的 RoBERTa Tokenizer 处理
        try:
            from transformers import AutoTokenizer

            # 加载 RoBERTa tokenizer
            tokenizer = AutoTokenizer.from_pretrained("roberta-base")
            print("Tokenizer loaded. Verifying Sequence Length = 180 constraint...")
        except:
            tokenizer = None
            print("Transformers not installed. Skipping tokenizer check.")

        for batch in dataloader:
            # 1. 检查 Visual 维度
            print(f"Visual Input Shape (ViT): {batch['image'].shape}")  # 应为 [B, 3, 224, 224]

            # 2. 检查 Textual 维度 (验证 180 长度)
            captions = batch['caption']
            print(f"Raw Caption Example: {captions[0][:50]}...")

            if tokenizer:
                # 您的要求: "encode it to form a fixed-length tensor (our sequence length is 180)"
                encoded_text = tokenizer(
                    list(captions),
                    return_tensors='pt',
                    padding='max_length',  # 强制填充
                    truncation=True,  # 强制截断
                    max_length=180  # <--- 您的硬性要求
                )

                input_ids = encoded_text['input_ids']
                print(f"Textual Input Shape (RoBERTa): {input_ids.shape}")  # 应为 [B, 180]

                if input_ids.shape[1] == 180:
                    print("✅ Sequence length check PASSED (180).")
                else:
                    print(f"❌ Sequence length check FAILED. Got {input_ids.shape[1]}")

            print(f"Labels: {batch['label']}")
            break
    else:
        print("Dataset is empty. Please check your path and ensure '句子_*.txt' exists.")