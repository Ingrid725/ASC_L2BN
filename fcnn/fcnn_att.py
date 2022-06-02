import keras
from keras.layers import Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, Dense
from keras.layers import Input, Dropout, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Model
from attention_layer import channel_attention
from keras import backend as K

class L2BN2d(BatchNormalization):
    '''
    this is for the Convolutional neural network
    '''
    def __init__(self, center_=True, scale_=True):
        super(L2BN2d, self).__init__(center=center_, scale=scale_)
        
    def forward(self, input):
        '''
        input: (N, C, H, W), N is the batchsize; C is the dimension or Channel nums.
        '''
        inshape = input.shape
        input_ = F.normalize(input.reshape(inshape[0], -1)).reshape(inshape)
        
        return super(L2BN2d, self).forward(input_)

# network definition
def resnet_layer(inputs,l2bn=False,num_filters=16,kernel_size=3,strides=1,learn_bn = True,wd=1e-4,use_relu=True):
    x = inputs
    x = Conv2D(num_filters,kernel_size=kernel_size,strides=strides,padding='valid',kernel_initializer='he_normal',
                  kernel_regularizer=l2(wd),use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

def conv_layer1(inputs, l2bn=False, num_channels=6, num_filters=14, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size1 = [5, 5]
    kernel_size2 = [3, 3]
    strides1 = [2, 2]
    strides2 = [1, 1]
    x = inputs
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    x = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size1, strides=strides1,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        x = Activation('relu')(x)

    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size2, strides=strides2,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2),padding='valid')(x)
    return x


def conv_layer2(inputs, l2bn=False, num_channels=6, num_filters=28, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters * num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x

def conv_layer3(inputs, l2bn=False, num_channels=6, num_filters=56, learn_bn=True, wd=1e-4, use_relu=True):
    kernel_size = [3, 3]
    strides = [1, 1]
    x = inputs
    # 1
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    #2
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    # 3
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    #4
    x = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(x)
    x = Conv2D(num_filters*num_channels, kernel_size=kernel_size, strides=strides,
               padding='valid', kernel_initializer='he_normal',
               kernel_regularizer=l2(wd), use_bias=False)(x)
    if not l2bn:
        x = BatchNormalization(center=learn_bn, scale=learn_bn)(x)
    else:
        x = L2BN2d(center_=learn_bn, scale_=learn_bn)(x)
    if use_relu:
        #x = Activation(relu6)(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), padding='valid')(x)
    return x



def model_fcnn(num_classes, l2bn=False, input_shape=[None, 128, 6], num_filters=[24, 48, 96], wd=1e-3):
    inputs = Input(shape=input_shape)
    ConvPath1 = conv_layer1(inputs=inputs,
                            l2bn=l2bn,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[0],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath2 = conv_layer2(inputs=ConvPath1,
                            l2bn=l2bn,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[1],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)
    ConvPath3 = conv_layer3(inputs=ConvPath2,
                            l2bn=l2bn,
                            num_channels=input_shape[-1],
                            num_filters=num_filters[2],
                            learn_bn=True,
                            wd=wd,
                            use_relu=True)

    # output layers after last sum
    OutputPath = resnet_layer(inputs=ConvPath3,
                              l2bn=l2bn,
                              num_filters=num_classes,
                              strides=1,
                              kernel_size=1,
                              learn_bn=False,
                              wd=wd,
                              use_relu=True)

    OutputPath = BatchNormalization(center=False, scale=False)(OutputPath)
    OutputPath = channel_attention(OutputPath, ratio=2)
    OutputPath = GlobalAveragePooling2D()(OutputPath)
    OutputPath = Activation('softmax')(OutputPath)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=OutputPath)
    return model
    