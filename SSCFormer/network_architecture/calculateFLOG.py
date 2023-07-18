import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ptflops import get_model_complexity_info
from SSCFormer.network_architecture.SSCFormer_heart import SSCFormer

# ACDC
# model = SSCFormer(in_channels=1,
#                    out_channels=4,
#                    feature_size=16,
#                    num_heads=4,
#                    depths=[3, 3, 3, 3],
#                    dims=[32, 64, 128, 256],
#                    do_ds=True,
#                    )
# tumor
# model = SSCFormer(in_channels=4,
#                    out_channels=4,
#                    img_size=[128, 128, 128],
#                    feature_size=16,
#                    num_heads=4,
#                    depths=[3, 3, 3, 3],
#                    dims=[32, 64, 128, 256],
#                    do_ds=True,
#                    )
# synapse
model = SSCFormer(in_channels=1,
                   out_channels=4,
                   img_size=[64, 128, 128],
                   feature_size=16,
                   num_heads=4,
                   depths=[3, 3, 3, 3],
                   dims=[32, 64, 128, 256],
                   do_ds=True,
                   )
# model = SSCFormer(crop_size=[64, 128, 128],
#                  embedding_dim=192,
#                  input_channels=1,
#                  num_classes=14,
#                  conv_op=nn.Conv3d,
#                  depths=[2, 2, 2, 2],
#                  num_heads=[6, 8, 24, 48],
#                  patch_size=[2, 4, 4],
#                  window_size=[4,4,8,4],
#                  deep_supervision=True)
flops, params = get_model_complexity_info(model, (1, 64, 128, 128), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)
