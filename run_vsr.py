import os
import cv2
import torch
import numpy as np
from mmagic.apis import MMagicInferencer
from mmengine import mkdir_or_exist

# 1. 使用正确的绝对路径
video_path = '/lustre/home/acct-cwj/cwj-user1/BasicVsr++/resources/input/video_interpolation/DJI_20260113111034_0008_D.MP4'
result_out_dir = '/lustre/home/acct-cwj/cwj-user1/BasicVsr++/resources/output/video_super_resolution/111.MP4'

print(f"视频路径: {video_path}")
print(f"输出路径: {result_out_dir}")

# 2. 验证视频文件
if not os.path.exists(video_path):
    print(f"错误：视频文件不存在！")
    exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("错误：无法打开视频文件！")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"\n视频信息:")
print(f"  - 原始分辨率: {width}x{height}")
print(f"  - 总帧数: {frame_count}")
print(f"  - FPS: {fps:.2f}")

# 3. 检查是否需要降低分辨率（3072x3072太大）
if width > 1920 or height > 1080:
    print(f"\n警告：视频分辨率 {width}x{height} 可能太大！")
    print("建议降低分辨率处理...")
    
    # 创建降低分辨率的临时视频
    import tempfile
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, 'downscaled_video.mp4')
    
    # 降低分辨率到960x960（保持宽高比）
    target_size = 2048
    scale_factor = target_size / max(width, height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    print(f"  - 降低分辨率到: {new_width}x{new_height}")
    
    # 使用OpenCV重新编码视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))
    
    frame_idx = 0
    test_frames = 30  # 只处理前30帧测试
    
    while frame_idx < test_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 调整大小
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        out.write(resized_frame)
        
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"  已处理 {frame_idx}/{test_frames} 帧")
    
    cap.release()
    out.release()
    
    print(f"临时视频已保存到: {temp_video_path}")
    video_path = temp_video_path  # 使用降分辨率后的视频
    width, height = new_width, new_height
    frame_count = test_frames
else:
    cap.release()

# 4. 确保输出目录存在
os.makedirs(os.path.dirname(result_out_dir), exist_ok=True)
print(f"\n输出目录已创建: {os.path.dirname(result_out_dir)}")

# 5. 加载模型
print("\n正在加载BASICVSR模型...")
try:
    # 禁用警告
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建inferencer - 使用更稳定的配置
    editor = MMagicInferencer(
        'basicvsr',
        model_setting=1,
        device=device
    )
    
    print("模型加载成功！")
    
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 6. 先处理少量帧进行测试
print(f"\n开始测试处理（分辨率: {width}x{height}, 帧数: {min(30, frame_count)}）...")
try:
    # 创建测试输出路径
    test_out_dir = result_out_dir.replace('.MP4', '_test.MP4')
    
    # 使用更简单的参数
    results = editor.infer(
        video=video_path, 
        result_out_dir=test_out_dir,
        fps=fps,
        start_idx=0,
        end_idx=min(30, frame_count),
        batch_size=1,  # 设置为1避免内存问题
        max_seq_len=10  # 限制序列长度
    )
    
    print(f"测试处理完成！输出保存到: {test_out_dir}")
    
    # 检查输出文件
    if os.path.exists(test_out_dir):
        output_size = os.path.getsize(test_out_dir) / (1024*1024)
        print(f"输出文件大小: {output_size:.2f} MB")
        
        # 验证输出视频
        cap_out = cv2.VideoCapture(test_out_dir)
        if cap_out.isOpened():
            out_frame_count = int(cap_out.get(cv2.CAP_PROP_FRAME_COUNT))
            out_fps = cap_out.get(cv2.CAP_PROP_FPS)
            out_width = int(cap_out.get(cv2.CAP_PROP_FRAME_WIDTH))
            out_height = int(cap_out.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_out.release()
            
            print(f"\n输出视频信息:")
            print(f"  - 分辨率: {out_width}x{out_height}")
            print(f"  - 帧数: {out_frame_count}")
            print(f"  - FPS: {out_fps:.2f}")
            
            # 判断是否进行了超分
            scale_factor = out_width / width
            if scale_factor > 1.5:
                print(f"  ✓ 超分辨率成功！放大了 {scale_factor:.1f} 倍")
            elif scale_factor > 1.0:
                print(f"  ✓ 轻微超分辨率，放大了 {scale_factor:.1f} 倍")
            else:
                print(f"  ⚠ 分辨率未改变或变小，可能是模型配置问题")
        else:
            print("警告：无法验证输出视频")
    
except Exception as e:
    print(f"\n处理过程中出错: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试使用图像序列方式
    print("\n尝试手动图像序列方式...")
    try:
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        frames_dir = os.path.join(temp_dir, 'frames')
        os.makedirs(frames_dir, exist_ok=True)
        
        print(f"临时目录: {temp_dir}")
        
        # 提取视频帧
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        max_frames = 10  # 只处理10帧
        
        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 保存帧
            frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.png')
            cv2.imwrite(frame_path, frame)
            print(f"保存帧 {frame_idx}: {frame.shape}")
            frame_idx += 1
        
        cap.release()
        
        if frame_idx == 0:
            print("错误：没有提取到任何帧！")
        else:
            print(f"成功提取 {frame_idx} 帧")
            
            # 处理图像序列
            frame_files = sorted([
                os.path.join(frames_dir, f) 
                for f in os.listdir(frames_dir) 
                if f.endswith('.png')
            ])
            
            if frame_files:
                print(f"开始处理 {len(frame_files)} 张图像...")
                results = editor.infer(
                    img=frame_files,
                    result_out_dir=frames_dir + '_processed',
                    fps=fps
                )
                print("图像处理完成！")
                
                # 如果需要，可以将处理后的图像合成为视频
                processed_dir = frames_dir + '_processed'
                if os.path.exists(processed_dir):
                    processed_files = sorted([
                        os.path.join(processed_dir, f) 
                        for f in os.listdir(processed_dir) 
                        if f.endswith('.png')
                    ])
                    
                    if processed_files:
                        # 创建输出视频
                        out_video_path = result_out_dir.replace('.MP4', '_from_frames.MP4')
                        first_frame = cv2.imread(processed_files[0])
                        h, w = first_frame.shape[:2]
                        
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
                        
                        for frame_file in processed_files:
                            frame = cv2.imread(frame_file)
                            out.write(frame)
                        
                        out.release()
                        print(f"视频已保存到: {out_video_path}")
                        
    except Exception as e2:
        print(f"图像序列方式也失败: {e2}")
        traceback.print_exc()