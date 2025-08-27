import sys
import os
import time
import hashlib
import re
import numpy as np
from scipy.io import wavfile
import sounddevice as sd

# 全局变量
callsign = None
last_sent_packet = None
sample_rate = 44100
symbol_duration = 0.03
carrier_freq = 256
protocol_version = "V2.0"  # 协议版本

# 初始化
def init():
    global callsign
    if not os.path.exists('TR.log'):
        with open('TR.log', 'w') as f:
            pass
    if callsign is None:
        callsign = input("请输入您的呼号: ").strip().upper()
        print(f"呼号 {callsign} 已设置")

# 4QAM调制
def qam4_modulate(bits):
    # 4QAM映射表
    symbol_map = {
        '00': (1, 0),
        '01': (0, 1),
        '10': (-1, 0),
        '11': (0, -1)
    }
    
    # 补零使长度为偶数
    if len(bits) % 2 != 0:
        bits += '0'
    
    symbols = []
    for i in range(0, len(bits), 2):
        symbol = bits[i:i+2]
        symbols.append(symbol_map[symbol])
    
    return symbols

# 生成调制信号
def generate_signal(symbols):
    samples_per_symbol = int(sample_rate * symbol_duration)
    t = np.linspace(0, symbol_duration, samples_per_symbol, endpoint=False)
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    
    signal = np.array([])
    for i, q in symbols:
        signal = np.concatenate((signal, i * carrier_i + q * carrier_q))
    
    # 归一化
    signal = signal / np.max(np.abs(signal))
    return signal

# 字符串转二进制
def str_to_bin(s):
    return ''.join(f'{ord(c):08b}' for c in s)

# 二进制转字符串
def bin_to_str(b):
    return ''.join(chr(int(b[i:i+8], 2)) for i in range(0, len(b), 8))

# 构建数据包
def build_packet(message, msg_type='001'):
    global last_sent_packet
    
    # 分割长消息
    if len(message) > 50:
        packets = []
        for i in range(0, len(message), 50):
            chunk = message[i:i+50]
            packets.append(build_packet(chunk, msg_type))
        return packets
    
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    
    # 构建控制字段 (保留为空)
    control = ""
    
    # 构建原始数据包
    raw_packet = f"\\StStStStSt {protocol_version} {callsign} {timestamp} {msg_type} \"{message}\" \"{control}\""
    
    # 计算MD5
    md5 = hashlib.md5(raw_packet.encode('utf-8')).hexdigest()[:8]
    
    # 完整数据包
    full_packet = f"{raw_packet} {md5} \\OrOrOrOrOr"
    
    # 保存到最后发送的数据包
    last_sent_packet = full_packet
    
    # 记录到日志
    with open('TR.log', 'a') as f:
        f.write(full_packet + '\n')
    
    return full_packet

# 编码消息到FLAC
def encode_to_flac(message, output_file):
    packet = build_packet(message)
    
    if isinstance(packet, list):
        for i, p in enumerate(packet):
            current_output = output_file.replace('.flac', f'_{i+1}.flac')
            _encode_single_packet(p, current_output)
        return
    
    _encode_single_packet(packet, output_file)

def _encode_single_packet(packet, output_file):
    # 字符串转二进制
    binary = str_to_bin(packet)
    
    # 4QAM调制
    symbols = qam4_modulate(binary)
    
    # 生成信号
    signal = generate_signal(symbols)
    
    # 保存为FLAC
    wavfile.write(output_file, sample_rate, (signal * 32767).astype(np.int16))
    print(f"已编码消息到 {output_file}")

# 解码FLAC到消息
def decode_from_flac(input_file):
    # 读取FLAC文件
    rate, data = wavfile.read(input_file)
    
    if rate != sample_rate:
        print("警告: 采样率不匹配，可能影响解码结果")
    
    # 解调信号
    symbols = demodulate_signal(data)
    
    # 符号到二进制
    binary = symbols_to_bits(symbols)
    
    # 二进制到字符串
    message = bin_to_str(binary)
    
    # 解析数据包
    parsed = parse_packet(message)
    
    if parsed is None:
        print("解码失败: 无效的数据包格式")
        return None
    
    version, callsign, timestamp, msg_type, content, control, md5 = parsed
    
    # 检查协议版本
    if version != protocol_version:
        print(f"你的内核版本不支持解码当前版本消息，请下载{version}版（不是看内核版本号，是看协议版本号）")
        return None
    
    # 验证MD5
    raw_part = message.split(md5)[0].strip()
    expected_md5 = hashlib.md5(raw_part.encode('utf-8')).hexdigest()[:8]
    if expected_md5 != md5:
        print(f"警告: MD5校验失败 (预期: {expected_md5}, 实际: {md5})")
        # 构建重传数据包
        if last_sent_packet is not None:
            retransmit_packet = build_retransmit_packet()
            print("已自动构建重传数据包")
            return retransmit_packet
    
    print(f"解码结果: 协议版本={version}, 呼号={callsign}, 时间={timestamp}, 类型={msg_type}, 内容={content}, 控制={control}")
    return message

# 构建重传数据包
def build_retransmit_packet():
    if last_sent_packet is None:
        print("错误: 没有上一个发送的数据包可用于重传")
        return None
    
    # 解析最后一个数据包
    match = re.match(r'\\StStStStSt (.+?) (.+?) (.+?) (.+?) "(.+?)" "(.+?)" (.+?) \\OrOrOrOrOr', last_sent_packet)
    if not match:
        print("错误: 无法解析上一个数据包")
        return None
    
    version, callsign, old_timestamp, msg_type, content, control, md5 = match.groups()
    
    # 构建重传数据包 (使用002作为重传类型)
    retransmit_packet = build_packet(content, '002')
    return retransmit_packet

# 解调信号
def demodulate_signal(signal):
    # 归一化信号
    signal = signal / np.max(np.abs(signal))
    
    # 计算每个符号的样本数
    samples_per_symbol = int(sample_rate * symbol_duration)
    num_symbols = len(signal) // samples_per_symbol
    
    # 解调
    symbols = []
    t = np.linspace(0, symbol_duration, samples_per_symbol, endpoint=False)
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    
    for i in range(num_symbols):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        symbol_signal = signal[start:end]
        
        # 计算I和Q分量
        i_component = np.sum(symbol_signal * carrier_i) / samples_per_symbol
        q_component = np.sum(symbol_signal * carrier_q) / samples_per_symbol
        
        # 判决
        if i_component > 0.5 and abs(q_component) < 0.5:
            symbols.append((1, 0))  # 00
        elif q_component > 0.5 and abs(i_component) < 0.5:
            symbols.append((0, 1))  # 01
        elif i_component < -0.5 and abs(q_component) < 0.5:
            symbols.append((-1, 0))  # 10
        elif q_component < -0.5 and abs(i_component) < 0.5:
            symbols.append((0, -1))  # 11
        else:
            # 无法判决，使用最近邻
            distances = [
                ((i_component-1)**2 + q_component**2),   # 00
                (i_component**2 + (q_component-1)**2),   # 01
                ((i_component+1)**2 + q_component**2),   # 10
                (i_component**2 + (q_component+1)**2)    # 11
            ]
            min_idx = np.argmin(distances)
            if min_idx == 0:
                symbols.append((1, 0))
            elif min_idx == 1:
                symbols.append((0, 1))
            elif min_idx == 2:
                symbols.append((-1, 0))
            else:
                symbols.append((0, -1))
    
    return symbols

# 符号到二进制
def symbols_to_bits(symbols):
    symbol_to_bits = {
        (1, 0): '00',
        (0, 1): '01',
        (-1, 0): '10',
        (0, -1): '11'
    }
    
    bits = ''
    for symbol in symbols:
        bits += symbol_to_bits.get(tuple(symbol), '00')  # 默认00
    
    return bits

# 解析数据包
def parse_packet(packet):
    # 检查开始和结束标记
    if not packet.startswith('\\StStStStSt') or not packet.endswith('\\OrOrOrOrOr'):
        print("错误: 数据包缺少开始或结束标记")
        return None
    
    # 提取MD5
    md5_match = re.search(r'([a-f0-9]{8}) \\OrOrOrOrOr$', packet)
    if not md5_match:
        print("错误: 无法提取MD5校验和")
        return None
    md5 = md5_match.group(1)
    
    # 验证MD5
    raw_part = packet.split(md5)[0].strip()
    expected_md5 = hashlib.md5(raw_part.encode('utf-8')).hexdigest()[:8]
    if expected_md5 != md5:
        print(f"警告: MD5校验失败 (预期: {expected_md5}, 实际: {md5})")
    
    # 解析字段 - 更新正则表达式以包含协议版本
    pattern = r'\\StStStStSt (.+?) (.+?) (.+?) (.+?) "(.+?)" "(.+?)"'
    match = re.match(pattern, packet)
    if not match:
        print("错误: 无法解析数据包字段")
        return None
    
    version, callsign, timestamp, msg_type, content, control = match.groups()
    
    return version, callsign, timestamp, msg_type, content, control, md5

# 主函数
def main():
    init()
    
    if len(sys.argv) < 2:
        print("用法: python3 FAITCC.py input.flac")
        print("或: python3 FAITCC.py \"消息内容\" output.flac")
        return
    
    if sys.argv[1].endswith('.flac'):
        # 解码模式
        input_file = sys.argv[1]
        if not os.path.exists(input_file):
            print(f"错误: 文件 {input_file} 不存在")
            return
        
        result = decode_from_flac(input_file)
        if result is None:
            print("解码失败")
        
    else:
        # 编码模式
        if len(sys.argv) < 3:
            print("错误: 需要指定输出文件名")
            print("用法: python3 FAITCC.py \"消息内容\" output.flac")
            return
        
        message = sys.argv[1]
        output_file = sys.argv[2]
        
        if not output_file.endswith('.flac'):
            output_file += '.flac'
        
        encode_to_flac(message, output_file)
        print(f"消息已编码到 {output_file}")

if __name__ == "__main__":
    print(f"欢迎使用 FAITCC(four QAM anti-interference text communication codec)！\n内核版本：V2.0 协议版本：{protocol_version} ©2025 童顺\n你可以用任何方式使用，分发，研究，改进，回馈本软件，但是必须保留版权声明 ")
    main()
