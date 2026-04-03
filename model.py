import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class model:
    def __init__(self):
        pass
    def depthwiseconv2d(self, input, weight_q, stride, scale_in, scale_out, scale_weight, bias):
        # input <1x49x10x1> <1x25x5x64>
        # kernel <1x10x4x64> <1x3x3x64>
        # steps and domain transform: convolution(uint8), add bias(int32), quantization(uint8), activation(uint8)
        input = input.astype(np.int32)
        kernel = np.load(weight_q).astype(np.int32)
        input_channel = input.shape[3]
        output = []
        if input_channel == 1 and stride == 2:
            pad_input = np.zeros((4+49+5, 1+10+1))
            pad_input[4:53, 1:11] = input[0, :, :, 0]
            win = sliding_window_view(pad_input, (10, 4))
            win = win[::2, ::2]
            for i in range(64):
                output_i = np.zeros((25, 5))
                for x in range(5):
                    for y in range(25):
                        output_i[y, x] = np.sum((win[y, x, :, :]-128) * (kernel[:, :, :, i]-128))
                output.append(output_i)
            output = np.array(output).reshape(1, 25, 5, 64)
        elif input_channel == 64 and stride == 1:
            pad_input = np.zeros((1+25+1, 1+5+1, 64))
            for i in range(64):
                pad_input[1:26, 1:6, i] = input[0, :, :, i]
                win = sliding_window_view(pad_input[:, :, i], (3, 3))
                output_i = np.zeros((25, 5))
                for x in range(5):
                    for y in range(25):
                        output_i[y, x] = np.sum((win[y, x, :, :]-128) * (kernel[0, :, :, i]-128))
                output.append(output_i)
            output = np.array(output).reshape(1, 25, 5, 64)
        output = self.add_bias(output, bias)
        output = self.quantization(output*scale_in*scale_weight, scale_out)
        return self.Relu(output)
    
    def conv2d(self, input, weight_q, scale_in, scale_out, scale_weight, bias): # 1x1
        # input <1x25x5x64>
        # kernel <64x1x1x64>
        # steps and domain transform: convolution(uint8), add bias(int32), quantization(uint8), activation(uint8)
        kernel = np.load(weight_q).astype(np.int32)
        input = input.astype(np.int32)
        output = np.zeros((1, 25, 5, 64))
        for i in range(64):
            for y in range(25):
                for x in range(5):
                    output[0, y, x, i] = np.sum((kernel[:, 0, 0, i]-128) * (input[0, y, x, :] - 128))
        output = self.add_bias(output, bias)
        output = self.quantization(output*scale_in*scale_weight, scale_out)
        return self.Relu(output)
    
    def globalaveragepool2d(self, input):
        output = np.zeros((1, 1, 1, 64))
        for i in range(input.shape[3]):
            x = input[0, :, :, i].astype(np.int32)
            x = x - 128
            avg = np.round(np.mean(x)).astype(np.int32)
            q = avg + 128
            output[0, 0, 0, i] = np.clip(q, 0, 255)
        return output
    
    def fullyconnected(self, input, weight_q, scale_in, scale_out, scale_weight, bias):
        weight = np.load(weight_q).astype(np.int32)
        input = input.astype(np.int32)
        output = (input.reshape(1, 64)-128) @ (weight.transpose(1, 0)-128)
        output = self.add_bias(output, bias)
        output = self.quantization(output*scale_in*scale_weight, scale_out)
        return self.Relu(output)
    
    def add_bias(self, input, bias):
        bias_ = np.load(bias)
        output = input + bias_
        return output

    def quantization(self, real, scale, zero=128):
        q = np.round(real / scale) + zero
        q = np.clip(q, 0, 255)
        return q.astype(np.uint8)

    def Relu(self, x):
        # uint8 domain
        return np.maximum(128, x)
    
    
    def prediction(self, mfcc_in):  # 49x10
        input_q = self.quantization(mfcc_in, scale=1.937271237373352, zero=128)
        ###downsample
        downsample = self.depthwiseconv2d(input_q, "weights\DS-CNN_dw0.npy", stride=2, scale_in=1.937271237373352, scale_out=0.10932768881320953,
                                          scale_weight=0.0007880174671299756, bias="weights\DS-CNN_dw0_bias.npy")
        
        ###layer111
        layer1_dw = self.depthwiseconv2d(downsample, "weights\DS-CNN_dw1.npy", stride=1, scale_in=0.10932768881320953, scale_out=0.124430350959301,
                                         scale_weight=0.02164473570883274, bias="weights\DS-CNN_dw1_bias.npy")
        layer1_pw = self.conv2d(layer1_dw, "weights\DS-CNN_co1.npy", scale_in=0.124430350959301, scale_out=0.10584013164043427,
                                scale_weight=0.005275495816022158, bias="weights\DS-CNN_co1_bias.npy")

        ###layer222
        layer2_dw = self.depthwiseconv2d(layer1_pw, "weights\DS-CNN_dw2.npy", stride=1, scale_in=0.10584013164043427, scale_out=0.1091616228222847,
                                         scale_weight=0.013021008111536503, bias="weights\DS-CNN_dw2_bias.npy")
        layer2_pw = self.conv2d(layer2_dw, "weights\DS-CNN_co2.npy", scale_in=0.1091616228222847, scale_out=0.07970251142978668,
                                scale_weight=0.005137702450156212, bias="weights\DS-CNN_co2_bias.npy")
        ###layer333
        layer3_dw = self.depthwiseconv2d(layer2_pw, "weights\DS-CNN_dw3.npy", stride=1, scale_in=0.07970251142978668, scale_out=0.09610754996538162,
                                         scale_weight=0.012005737982690334, bias="weights\DS-CNN_dw3_bias.npy")
        layer3_pw = self.conv2d(layer3_dw, "weights\DS-CNN_co3.npy", scale_in=0.09610754996538162, scale_out=0.07651054859161377,
                                scale_weight=0.005375369917601347, bias="weights\DS-CNN_co3_bias.npy")
        ###layer444
        layer4_dw = self.depthwiseconv2d(layer3_pw, "weights\DS-CNN_dw4.npy", stride=1, scale_in=0.07651054859161377, scale_out=0.11575359851121902,
                                         scale_weight=0.009517693892121315, bias="weights\DS-CNN_dw4_bias.npy")

        layer4_pw = self.conv2d(layer4_dw, "weights\DS-CNN_co4.npy", scale_in=0.11575359851121902, scale_out=0.08714716881513596,
                                scale_weight=0.007161697372794151, bias="weights\DS-CNN_co4_bias.npy")

        ###
        layer_avgpool = self.globalaveragepool2d(layer4_pw)

        layer_fc = self.fullyconnected(layer_avgpool, "weights\DS-CNN_fc.npy", scale_in=0.08714716881513596, scale_out=0.23083342611789703,
                                       scale_weight=0.009661822579801083, bias="weights\DS-CNN_fc_bias.npy")
        # print("min/max:", np.min(ly4_pw_act), np.max(ly4_pw_act))
        return layer_fc







