
if __name__=="eznet_keras.test":
    from .utils import *
    from .models import *
else:
    from utils import *
    from models import *

def testcalc_image_size():
    size = calc_image_size(size_in=[28,28,28], kernel_size=3, padding='valid', stride=2, dilation=1)
    print(size)

def testgenerate_geometric_array():
    array = generate_geometric_array(init=256, count=4, direction='up', powers_of_two=True)
    print(array)
    
def testgenerate_array_for_hparam():
    array = generate_array_for_hparam(
        [1,2], count_if_not_list=4, 
        hparam_name='parameter', count_if_not_list_name='its count',
        check_auto=True, init_for_auto=2, powers_of_two_if_auto=True,
        direction_if_auto='up')
    print(array)

def test_dense_block():
    dense_block = Dense_Block([256], 128, 'relu', None, 'BatchNormalization', 'before', None, 0.1, tf.keras.regularizers.L2(0.0001))
    print(dense_block)
    print(dense_block.summary())
    x = np.random.randn(32,256)
    y = dense_block(x)
    print("Input shape:  ", x.shape)
    print("Output shape: ", y.shape)

def test_conv_block():
    conv_block = Conv_Block([28,28,3], 32, conv_dim=2, input_image=[28,28], conv_kernel_size=3, conv_padding='valid', conv_stride=1, 
                            conv_dilation=1, conv_params=None, 
                            conv_activation='relu', conv_activation_params=None, norm_layer_position='before', norm_layer_type='BatchNormalization', 
                            norm_layer_params=None, 
                            pool_type='Max', pool_kernel_size=2, pool_padding='valid', pool_stride=1, pool_params=None, dropout=0.1, min_image_dim=8,
                            kernel_regularizer=tf.keras.regularizers.L2(0.0001))
    print(conv_block)
    print(conv_block.summary())
    x = np.random.randn(32,28,28,3)
    y = conv_block(x)
    print("Input shape:  ", x.shape)
    print("Output shape: ", y.shape)
    
def test_conv_network():
    test_keras_model_class(Conv_Network)

def test_recurrent_network():
    test_keras_model_class(Recurrent_Network)
    
def test_ann_network():
    test_keras_model_class(ANN)




if __name__ == '__main__':
    
    # testcalc_image_size()
    # testgenerate_geometric_array()
    # testgenerate_array_for_hparam()
    
    # test_dense_block()
    # test_conv_block()
    
    # test_conv_network()
    # test_recurrent_network()
    # test_ann_network()
    
    
    pass