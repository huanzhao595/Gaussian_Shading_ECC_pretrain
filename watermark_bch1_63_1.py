import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes
import galois

# # bch(63,1), without disturb
# !python run_gaussian_shading.py \
#       --fpr 0.000001 \
#       --channel_copy 1 \
#       --hw_copy 8 \
#       --chacha 1\
#       --num 100 \
#       --reference_model ViT-g-14 \
#       --reference_model_pretrain laion2b_s12b_b42k 

# nohup python run_gaussian_shading.py --fpr 0.000001 --channel_copy 1 --hw_copy 8 --chacha 1 --num 2 --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k &




# nohup python run_gaussian_shading.py \
#   --fpr 0.000001 \
#   --channel_copy 1 \
#   --hw_copy 8 \
#   --chacha 1 \
#   --num 2 \
#   --reference_model ViT-g-14 \
#   --reference_model_pretrain laion2b_s12b_b42k \
#   > gaussian_shading.log 2>&1 &


# # bch(63,1), with disturb: [jpeg_ratio, random_crop_ratio, random_drop_ratio, 
                            #     gaussian_blur_r, median_blur_k, resize_ratio,
                            # gaussian_std, sp_prob, brightness_factor]
# !python run_gaussian_shading.py \
#       --gaussian_std 0.05 \
#       --fpr 0.000001 \
#       --channel_copy 1 \
#       --hw_copy 8 \
#       --chacha 1 \
#       --num 100 \
#       --reference_model ViT-g-14 \
#       --reference_model_pretrain laion2b_s12b_b42k 

class Gaussian_Shading_chacha:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.nonce = None
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        # self.threshold_1 = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.threshold = 1 if self.hw == 1 and self.ch == 1 else 51 // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def stream_key_encrypt(self, sd):
        self.key = get_random_bytes(32)
        self.nonce = get_random_bytes(12)
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)

        # scale_factor = sd.max()
        # normalized_sd = (sd / scale_factor).astype(np.uint8)
        packed_bits = np.packbits(sd)
        encrypted_bytes = cipher.encrypt(packed_bits.tobytes())
        encrypted_bits = np.unpackbits(np.frombuffer(encrypted_bytes, dtype=np.uint8))
        return encrypted_bits   #m_bit

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w_1(self):
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = self.stream_key_encrypt(sd.flatten().cpu().numpy())
        w = self.truncSampling(m)
        return w

    def stream_key_decrypt(self, reversed_m):
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)

        encrypted_bytes = np.packbits(reversed_m)
        decrypted_bytes = cipher.decrypt(encrypted_bytes.tobytes())
        decrypted_bits = np.unpackbits(np.frombuffer(decrypted_bytes, dtype=np.uint8))
        # restored_sd = (decrypted_bits * scale_factor).astype(np.uint8)
        decrypted_tensor = torch.from_numpy(decrypted_bits).reshape(1, 4, 64, 64).to(torch.uint8) #decrypted_bits
        
        return decrypted_tensor.cuda()

    def diffusion_inverse_1(self,watermark_r):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_r, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark_1(self, reversed_w):
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct
        
    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

    def diffusion_inverse(self, watermark_r):
        """
        watermark_r: Tensor of shape (1, 4, 64, 51)
        """
        # 直接在重复维度投票 (dim=3 是重复51次)
        vote = torch.sum(watermark_r, dim=3).clone()  # shape: (1, 4, 64)
        
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        
        vote = vote.view(1, 4 // self.ch, self.hw, self.hw)  # e.g. (1, 4, 8, 8)
        return vote

    def encode_bch(self, raw_message, BCH_N, BCH_K):
        # 确保 raw_message 是 NumPy 数组
        if isinstance(raw_message, torch.Tensor):
            raw_message = raw_message.cpu().numpy()  # 转换为 NumPy 数组
        
        # Create BCH(63,1) code
        GF2 = galois.GF(2)
        BCH = galois.BCH(n=BCH_N, k=BCH_K)
        flat_bits = raw_message.flatten()
        assert flat_bits.size == 256, "原始数据比特数应为 256"

        # 每个比特作为一条消息，reshape 成 (256, 1)
        messages = flat_bits.reshape((-1, 1))
        # 转换到 GF(2)
        messages_GF = GF2(messages)
        
        # 构造 BCH(63, 1) 编码器
        # 对每个单比特消息编码，得到形状 (256, 63) 的码字（GF(2) 数组）
        codewords = BCH.encode(messages_GF)
        
        # 添加固定的 padding 位（0）：构造一个形状 (256, 1) 的全 0 GF(2) 数组，并水平拼接
        pad = GF2.Zeros((codewords.shape[0], 1))
        extended_codewords = np.hstack([codewords, pad])  # shape: (256, 64)
        
        # 转换为 numpy 数组后展平成一维，再重构为 (1, 4, 64, 64)
        encoded_bits = extended_codewords.view(np.ndarray)
        encoded_tensor = encoded_bits.reshape((1, 4, 64, 64))
        return encoded_tensor
    
    def decode_bch(self, encoded_tensor, BCH_N, BCH_K):   
        # Create BCH(63,1) code
        GF2 = galois.GF(2)
        BCH = galois.BCH(n=BCH_N, k=BCH_K)
        
        # 确保 raw_message 是 NumPy 数组
        if isinstance(encoded_tensor, torch.Tensor):
            encoded_tensor = encoded_tensor.cpu().numpy()  # 转换为 NumPy 数组
            
        # 拉平编码张量，共 1×4×64×64 = 16384 个比特
        flat = encoded_tensor.flatten()
        total_bits = flat.size
        assert total_bits == 16384, "总比特数应为 16384"
        
        # 重塑为 (256, 64)：每一行对应一个扩展码字
        extended_codewords = flat.reshape((-1, 64))
        # 去除扩展码字中最后的 padding 位（固定为 0），保留前 63 位作为 BCH 码字
        codewords = extended_codewords[:, :63]
        
        # 转换为 GF(2) 数组
        codewords_GF = GF2(codewords)
        
        # 构造 BCH(63, 1) 解码器，并解码恢复出每条消息（形状 (256, 1)）
        decoded_messages = BCH.decode(codewords_GF)
        
        # 转换为 numpy 数组后展平成一维，重塑为 (1, 4, 64, 1)
        decoded_bits = decoded_messages.view(np.ndarray).flatten()
        decoded_tensor = decoded_bits.reshape((1, 4, 64, 1))
        return decoded_tensor

    def print_data(self, data):
        df_original = pd.DataFrame(data[0, 0])  # Display first batch, first group
        # df_encoded = pd.DataFrame(padded_encoded_data[0, 0])  # Display first batch, first group
        # df_decoded = pd.DataFrame(decoded_data[0, 0])  # Display first batch, first group
        # df_restored = pd.DataFrame(restored[0, 0])
        
        print(df_original.head(), df_original.shape)

    def create_watermark_and_return_w(self):
        BCH_N = 63  # Codeword length
        BCH_K = 1
        original_shape = [1, 4 // self.ch, 64, 1]
        # repeat_shape = [1, 4 // self.ch, self.hw * self.hw, 51]
        encoded_shape = [1, 4 // self.ch, 64, 64]

        # Generate test data: (1,4,64,1) binary matrix (original information)
        self.watermark = np.random.randint(0, 2, (1, 4 // self.ch, 64, 1), dtype=np.uint8)
        
        encoded_message = self.encode_bch(torch.tensor(self.watermark, dtype=torch.uint8), BCH_N, BCH_K)
        print("self.watermark[0, 0]", self.watermark[0, 0])
        print("encoded_message[0, 0]", encoded_message[0, 0])
        m = self.stream_key_encrypt(encoded_message)
        w = self.truncSampling(m)
        return w

    def bit_acc(self, restored_watermark):
        if isinstance(restored_watermark, torch.Tensor):
            restored_watermark = restored_watermark.cpu().numpy()  # 转换为 NumPy 数组
        total_bits = self.watermark.size
        correct_bits = np.sum(self.watermark == restored_watermark)
        bit_accuracy = correct_bits / total_bits
        
        print(f"Bit Accuracy: {bit_accuracy:.4f}")
        return bit_accuracy
    
    def eval_watermark(self, reversed_w):
        BCH_N = 63  # Codeword length
        BCH_K = 1
        original_shape = [1, 4 // self.ch, 64, 1]
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        # reversed_sd = reversed_sd[:, :, :, :(64  - nsym)]  # 取前 10 列
        print("reversed_sd.shape: ", reversed_sd.shape)
        reversed_watermark = self.decode_bch(torch.tensor(reversed_sd, dtype=torch.uint8), BCH_N, BCH_K)
        print("reversed_watermark[0, 0]", reversed_watermark[0, 0])
        
        # restored_watermark = self.diffusion_inverse(torch.tensor(reversed_watermark, dtype=torch.uint8))
        
        # correct = (self.watermark == reversed_watermark).float().mean().item()
        correct = np.mean(self.watermark == reversed_watermark)
        print("correct", correct)
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        # hamming_dist = self.hamming_distance(raw_message, reversed_watermark)
        bit_accuracy = self.bit_acc(reversed_watermark)
        return bit_accuracy
    

class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None
        self.latentlength = 4 * 64 * 64
        self.marklength = self.latentlength//(self.ch * self.hw * self.hw)

        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2
        self.tp_onebit_count = 0
        self.tp_bits_count = 0
        self.tau_onebit = None
        self.tau_bits = None

        for i in range(self.marklength):
            fpr_onebit = betainc(i+1, self.marklength-i, 0.5)
            fpr_bits = betainc(i+1, self.marklength-i, 0.5) * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        z = np.zeros(self.latentlength)
        if isinstance(message, torch.Tensor):
            message = message.cpu().numpy()
        denominator = 2.0
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]
        for i in range(self.latentlength):
            dec_mes = reduce(lambda a, b: 2 * a + b, message[i : i + 1])
            dec_mes = int(dec_mes)
            z[i] = truncnorm.rvs(ppf[dec_mes], ppf[dec_mes + 1])
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w_2(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1,self.ch,self.hw,self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self,watermark_sd):
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw
        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark_2(self, reversed_m):
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)
        correct = (reversed_watermark == self.watermark).float().mean().item()
        if correct >= self.tau_onebit:
            self.tp_onebit_count = self.tp_onebit_count+1
        if correct >= self.tau_bits:
            self.tp_bits_count = self.tp_bits_count + 1
        return correct

    def get_tpr(self):
        return self.tp_onebit_count, self.tp_bits_count

    def encode_reed_solomon(self, raw_message, nsym):
        original_shape = (1,4,64,21)
        encoded_shape = (1, 4, 64, 64) 
        print(f"nsym type: {type(nsym)}, value: {nsym}")
        rs = reedsolo.RSCodec(nsym)

        # 确保 raw_message 是 NumPy 数组
        if isinstance(raw_message, torch.Tensor):
            raw_message = raw_message.cpu().numpy().astype(np.uint8)  # 转换为 NumPy 数组
        
        # 进行 Reed-Solomon 编码
        encoded_message = np.zeros(encoded_shape, dtype=np.uint8)
    
        for i in range(original_shape[0]):
            for j in range(original_shape[1]):
                for k in range(original_shape[2]):
                    encoded_data = rs.encode(raw_message[i, j, k, :].tobytes())  # 确保转换为 bytes
                    encoded_message[i, j, k, :] = np.frombuffer(encoded_data, dtype=np.uint8)
        # print("encoded_message", encoded_message.shape)
        return encoded_message
    
    def decode_reed_solomon(self, encoded_message, nsym):
        original_shape = (1,4,64,21)
        rs = reedsolo.RSCodec(nsym)

        # 确保 raw_message 是 NumPy 数组
        if isinstance(encoded_message, torch.Tensor):
            encoded_message = encoded_message.cpu().numpy().astype(np.uint8)  # 转换为 NumPy 数组
        
        # 进行 Reed-Solomon 解码
        decoded_message = np.zeros(original_shape, dtype=np.uint8)
        
        for i in range(original_shape[0]):
            for j in range(original_shape[1]):
                for k in range(original_shape[2]):
                    decoded_data = rs.decode(encoded_message[i, j, k, :].tobytes())  # 确保转换为 bytes、
                    if isinstance(decoded_data, tuple):  # 避免 tuple 错误
                        decoded_data = decoded_data[0]
        
                    decoded_message[i, j, k, :] = np.frombuffer(decoded_data, dtype=np.uint8)
        # print("decoded_message", decoded_message.shape)
        return torch.tensor(decoded_message, dtype=torch.uint8).cuda()
    
    # 计算 Hamming 距离（错误的比特数）
    def hamming_distance(self, arr1, arr2):
        return np.sum([bin(x ^ y).count('1') for x, y in zip(arr1.flatten(), arr2.flatten())])
    
    def bit_acc(self, original_shape, hamming_dist):
        total_bits = np.prod(original_shape) * 8
        bit_accuracy = (total_bits - hamming_dist) / total_bits
        return bit_accuracy

    def create_watermark_and_return_w_2(self):
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()
        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)
        m = ((sd + self.key) % 2).flatten().cpu().numpy()
        w = self.truncSampling(m)
        return w
        
    def create_watermark_and_return_w(self):
        nsym = 43
        original_shape = [1, 4 // self.ch, 64 , (64 - nsym)]  # (1, 4, 64, 32)
        encoded_shape = [1, 4 // self.ch, 64 , 64 ]  # (1, 4, 64, 64)
        
        # 生成水印
        raw_message = torch.randint(0, 2, original_shape).cuda()  # 确保是 half()
        
        # 进行 Reed-Solomon 编码
        encoded_message = self.encode_reed_solomon(torch.tensor(raw_message, dtype=torch.uint8), nsym)
        print("encoded_message.shape: ", encoded_message.shape)
        
        w = self.truncSampling(encoded_message)
        
        w = torch.tensor(w, dtype=torch.uint8).cuda()
        
        return w, raw_message

        
    def eval_watermark(self, raw_message, reversed_w):
        nsym = 43
        original_shape = [1, 4 // self.ch, 64 , (64  - nsym)] # (1, 4, 64, 32)
        reversed_m = (reversed_w > 0).int()
        reversed_sd = self.stream_key_decrypt(reversed_m.flatten().cpu().numpy())
        print("type(reversed_sd.flatten().cpu().numpy()): ", type(reversed_sd.flatten().cpu().numpy()))
        reversed_watermark = self.decode_reed_solomon(torch.tensor(reversed_sd, dtype=torch.uint8), nsym)

        hamming_dist = hamming_distance(raw_message, reversed_watermark)
        bit_accuracy = bit_acc(original_shape, hamming_dist)
        return hamming_dist, bit_accuracy


