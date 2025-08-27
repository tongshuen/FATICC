import sys
import os
import time
import hashlib
import re
import numpy as np
from scipy.io import wavfile
import sounddevice as sd
import shlex

# 全局变量
callsign = None
last_sent_packet = None
sample_rate = 44100
symbol_duration = 0.03
carrier_freq = 256
protocol_version = "V2.0"  # 协议版本
kernel_version = "V3.0"    # 内核版本

# 初始化
def init():
    global callsign
    if not os.path.exists('TR.log'):
        with open('TR.log', 'w') as f:
            pass
    if callsign is None:
        callsign = input("(FAITCC)→ 请输入您的呼号: ").strip().upper()
        print(f"(FAITCC)→ 呼号 {callsign} 已设置")

# 4QAM调制
def qam4_modulate(bits):
    symbol_map = {
        '00': (1, 0),
        '01': (0, 1),
        '10': (-1, 0),
        '11': (0, -1)
    }
    
    if len(bits) % 2 != 0:
        bits += '0'
    
    symbols = []
    for i in range(0, len(bits), 2):
        symbol = bits[i:i+2]
        symbols.append(symbol_map[symbol])
    
    return symbols

def generate_signal(symbols):
    samples_per_symbol = int(sample_rate * symbol_duration)
    t = np.linspace(0, symbol_duration, samples_per_symbol, endpoint=False)
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    
    signal = np.array([])
    for i, q极 symbols:
        signal = np.concatenate((signal, i * carrier_i + q * carrier_q))
    
    signal = signal / np.max(np.abs(signal))
    return signal

def str_to_bin(s):
    return ''.join(f'{ord(c):08b}' for c in s)

def bin_to_str(b):
    return ''.join(chr(int(b[i:i+8], 2)) for i in range(0, len(b), 8))

def build_packet(message, msg_type='001'):
    global last_sent_packet
    
    if len(message) > 50:
        packets = []
        for i in range(0, len(message), 50):
            chunk = message[i:i+50]
            packets.append(build_packet(chunk, msg_type))
        return packets
    
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    control = ""
    raw_packet = f"\\StStStStSt {protocol_version} {callsign} {timestamp} {msg_type} \"{message}\" \"{control}\""
    md5 = hashlib.md5(raw极packet.encode('utf-8')).hexdigest()[:8]
    full_packet = f"{raw_packet} {md5} \\OrOrOrOrOr"
    last_sent_packet = full_packet
    
    with open('TR.log', 'a') as f:
        f.write(full_packet + '\n')
    
    return full_packet

def encode_to_flac(message, output_file):
    packet = build_packet(message)
    
    if isinstance(packet, list):
        for i, p in enumerate(packet):
            base_name, ext = os.path.splitext(output_file)
            current_output = f"{base_name}_{i+1}{ext}"
            _encode_single_packet(p, current_output)
        return
    
    _encode_single_packet(packet, output_file)

def _encode_single_packet(packet, output_file):
    binary = str_to_bin(packet)
    symbols = qam4_modulate(binary)
    signal = generate_signal(symbols)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    wavfile.write(output_file, sample_rate, (signal * 32767).astype(np.int16))
    print(f"(FAITCC)→ 已编码到 {output_file}")

def decode_from_flac(input_file, output_target=None):
    if not os.path.exists(input_file):
        print(f"(FAITCC)→ 错误: 输入文件 {input_file} 不存在")
        return None
    
    try:
        rate, data = wavfile.read(input_file)
    except Exception as e:
        print(f"(FAITCC)→ 错误: 无法读取文件 {input_file}: {e}")
        return None
    
    if rate != sample_rate:
        print("(FAITCC)→ 警告: 采样率不匹配，可能影响解码结果")
    
    symbols = demodulate_signal(data)
    binary = symbols_to_bits(symbols)
    message = bin_to_str(binary)
    parsed = parse_packet(message)
    
    if parsed is None:
        print("(极AITCC)→ 解码失败: 无效的数据包格式")
        return None
    
    version, callsign, timestamp, msg_type, content, control, md5 = parsed
    
    if version != protocol_version:
        print(f"(FAITCC)→ 错误: 需要协议版本 {version} (当前 {protocol_version})")
        return None
    
    raw_part = message.split(md5)[0].strip()
    expected_md5 = hashlib.md5(raw_part.encode('utf-8')).hexdigest()[:8]
    if expected_md5 != md5:
        print(f"(FAITCC)→ 警告: MD5校验失败 (预期: {expected_md5}, 实际: {md5})")
        if last_sent_packet is not None:
            retransmit_packet = build_retransmit_packet()
            print("(FAITCC)→ 已自动构建重传数据包")
            return retransmit_packet
    
    if output_target == "/FAITCCTTY":
        print(f"(FAITCC)→ 解码结果:\n呼号: {callsign}\n时间: {timestamp}\n类型: {msg_type}\n内容: {content}")
        return content
    elif output_target:
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_target)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_target, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"(FAITCC)→ 解码结果已写入 {output_target}")
            return content
        except Exception as e:
            print(f"(FAITCC)→ 错误: 无法写入文件 {output_target}: {e}")
            return None
    else:
        return content

def build_retransmit_packet():
    if last_sent_packet is None:
        print("(FAITCC)→ 错误: 没有可用于重传的上一个数据包")
        return None
    
    match = re.match(r'\\StStStStSt (.+?) (.+?) (.+?) (.+?) "(.+?)" "(.+?)" (.+?) \\OrOrOrOrOr', last_sent_packet)
    if not match:
        print("(FAITCC)→ 错误: 无法解析上一个数据包")
        return None
    
    version, callsign, old_timestamp, msg_type, content, control, md5 = match.groups()
    retransmit_packet = build_packet(content, '002')
    return retransmit_packet

def demodulate_signal(signal):
    signal = signal / np.max(np.abs(signal))
    samples_per_symbol = int(sample_rate * symbol_duration)
    num_symbols = len(signal) // samples_per_symbol
    
    symbols = []
    t = np.linspace(0, symbol_duration, samples_per_symbol, endpoint=False)
    carrier_i = np.cos(2 * np.pi * carrier_freq * t)
    carrier_q = np.sin(2 * np.pi * carrier_freq * t)
    
    for i in range(num_symbols):
        start = i * samples_per_symbol
        end = start + samples_per_symbol
        symbol_signal = signal[start:end]
        
        i_component = np.sum(symbol_signal * carrier_i) / samples_per_symbol
        q_component = np.sum(symbol_signal * carrier_q) / samples_per_symbol
        
        if i_component > 0.5 and abs(q_component) < 0.5:
            symbols.append((1, 0))
        elif q_component > 0.5 and abs(i_component) < 0.5:
            symbols.append((0, 1))
        elif i_component < -0.5 and abs(q_component) < 0.5:
            symbols.append((-1, 0))
        elif q_component < -0.5 and abs(i_component) < 0.5:
            symbols.append((0, -1))
        else:
            distances = [
                ((i_component-1)**2 + q_component**2),
                (i_component**2 + (q_component-1)**2),
                ((i_component+1)**2 + q_component**2),
                (i_component**2 + (q_component+1)**2)
            ]
            min_idx = np.argmin(distances)
            symbols.append([(1, 0), (0, 1), (-1, 0), (0, -1)][min_idx])
    
    return symbols

def symbols_to_b极its(symbols):
    symbol_to_bits = {
        (1, 0): '00',
        (0, 1): '01',
        (-1, 0): '10',
        (0, -1): '11'
    }
    
    bits = ''
    for symbol in symbols:
        bits += symbol_to_bits.get(tuple(symbol), '00')
    
    return bits

def parse_packet(packet):
    if not packet.startswith('\\StStStStSt') or not packet.endswith('\\Or极OrOrOrOr'):
        print("(FAITCC)→ 错误: 数据包缺少开始/结束标记")
        return None
    
    md5_match = re.search(r'([a-f0-9]{8}) \\OrOrOrOrOr$', packet)
    if not md5_match:
        print("(FAITCC)→ 错误: 无法提取MD5校验和")
        return None
    md5 = md5_match.group(1)
    
    pattern = r'\\StStStStSt (.+?) (.极+?) (.+?) (.+?) "(.+?)" "(.+?)"'
    match = re.match(pattern, packet)
    if not match:
        print("(FAITCC)→ 错误: 无法解析数据包字段")
        return None
    
    version, callsign, timestamp, msg_type, content, control = match.groups()
    return version, callsign, timestamp, msg_type, content, control, md5

def process_command(cmd):
    try:
        # 使用shlex.split正确处理带引号的路径
        parts = shlex.split(cmd)
        if len(parts) != 2:
            print("(FAITCC)→ 错误: 需要两个参数 (输入 输出)")
            print("(FAITCC)→ 例如: \"/path/to/input.txt\" \"/path/to/output.flac\"")
            return
        
        input_arg, output_arg = parts
        
        # 编码: 输入是文本或文本文件，输出是.flac文件
        if not input_arg.lower().endswith('.flac'):
            # 检查输入是否是现有文件
            if os.path.exists(input_arg):
                try:
                    with open(input_arg, 'r', encoding='utf-8') as f:
                        content = f.read()
                    print(f"(FAITCC)→ 从文件 {input_arg} 读取内容")
                except Exception as e:
                    print(f"(FAITCC)→ 错误: 无法读取文件 {input_arg}: {e}")
                    return
            else:
                # 输入是直接文本
                content = input_arg
                print(f"(FAITCC)→ 编码文本内容")
            
            # 确保输出文件以.flac结尾
            if not output_arg.lower().endswith('.flac'):
                output_arg += '.flac'
            
            encode_to_flac(content, output_arg)
        
        # 解码: 输入是.flac文件，输出是文本文件或终端
        else:
            if output_arg == "/FAITCCTTY":
                decode_from_flac(input_arg, "/FAITCCTTY")
            else:
                # 确保输出文件以.txt结尾（如果不是特殊目标）
                if not output_arg.lower().endswith('.txt') and output_arg != "/FAITCCTTY":
                    output_arg += '.txt'
                decode_from_flac(input_arg, output_arg)
                
    except Exception as e:
        print(f"(FAITCC)→ 命令处理错误: {e}")

def interactive_shell():
    print(f"(FAITCC)→ FAITCC 交互模式 (内核版本 {kernel_version})")
    print("(FAITCC)→ 输入帮助命令: help")
    init()
    
    while True:
        try:
            cmd = input("(FAITCC)→ ").strip()
            if not cmd:
                continue
                
            if cmd.lower() in ['exit', 'quit']:
                print("(FAITCC)→ 73!")
                break
            elif cmd.lower() == 'help':
                print("""(FAITCC)→ 可用命令:
  "文本内容" /path/to/output.flac    # 编码文本到FLAC文件
  /path/to/input.txt /path/to/output.flac  # 编码文本文件到FLAC
  /path/to/input.flac /path/to/output.txt  # 解码FLAC到文本文件
  /path/to/input.flac /FAITCCTTY           # 解码FLAC到终端显示
  exit/quit                                # 退出程序
  
示例:
  "Hello World" /tmp/test.flac
  /home/user/message.txt /tmp/encoded.flac  
  /tmp/encoded.flac /home/user/decoded.txt
  /tmp/encoded.flac /FAITCCTTY""")
            else:
                process_command(cmd)
        except KeyboardInterrupt:
            print("\n(FAITCC)→ 使用 exit 或 quit 退出程序")
        except Exception as e:
            print(f"(FAITCC)→ 错误: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        interactive_shell()
    else:
        print("(FAITCC)→ 直接运行程序进入交互模式，无需参数")
