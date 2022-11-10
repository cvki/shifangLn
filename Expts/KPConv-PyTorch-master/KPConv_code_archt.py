# <bound method Module.named_parameters of KPCNN(
#   (block_ops): ModuleList(
#     (0): SimpleBlock(
#       (KPConv): KPConv(radius: 0.05, in_feat: 1, out_feat: 32)
#       (batch_norm): BatchNormBlock(in_feat: 32, momentum: 0.050, only_bias: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (1): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 32, out_feat: 16, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.05, in_feat: 16, out_feat: 16)
#       (batch_norm_conv): BatchNormBlock(in_feat: 16, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 16, out_feat: 64, BN: True, ReLU: False)
#       (unary_shortcut): UnaryBlock(in_feat: 32, out_feat: 64, BN: True, ReLU: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (2): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 64, out_feat: 16, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.05, in_feat: 16, out_feat: 16)
#       (batch_norm_conv): BatchNormBlock(in_feat: 16, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 16, out_feat: 64, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (3): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 64, out_feat: 32, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.10, in_feat: 32, out_feat: 32)
#       (batch_norm_conv): BatchNormBlock(in_feat: 32, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 32, out_feat: 128, BN: True, ReLU: False)
#       (unary_shortcut): UnaryBlock(in_feat: 64, out_feat: 128, BN: True, ReLU: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (4): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 128, out_feat: 32, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.10, in_feat: 32, out_feat: 32)
#       (batch_norm_conv): BatchNormBlock(in_feat: 32, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 32, out_feat: 128, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (5): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 128, out_feat: 32, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.10, in_feat: 32, out_feat: 32)
#       (batch_norm_conv): BatchNormBlock(in_feat: 32, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 32, out_feat: 128, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (6): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 128, out_feat: 64, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.20, in_feat: 64, out_feat: 64)
#       (batch_norm_conv): BatchNormBlock(in_feat: 64, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 64, out_feat: 256, BN: True, ReLU: False)
#       (unary_shortcut): UnaryBlock(in_feat: 128, out_feat: 256, BN: True, ReLU: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (7): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 256, out_feat: 64, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.20, in_feat: 64, out_feat: 64)
#       (batch_norm_conv): BatchNormBlock(in_feat: 64, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 64, out_feat: 256, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (8): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 256, out_feat: 64, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.20, in_feat: 64, out_feat: 64)
#       (batch_norm_conv): BatchNormBlock(in_feat: 64, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 64, out_feat: 256, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (9): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 256, out_feat: 128, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.40, in_feat: 128, out_feat: 128)
#       (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
#       (unary_shortcut): UnaryBlock(in_feat: 256, out_feat: 512, BN: True, ReLU: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (10): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 512, out_feat: 128, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.40, in_feat: 128, out_feat: 128)
#       (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (11): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 512, out_feat: 128, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.40, in_feat: 128, out_feat: 128)
#       (batch_norm_conv): BatchNormBlock(in_feat: 128, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 128, out_feat: 512, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (12): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 512, out_feat: 256, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.80, in_feat: 256, out_feat: 256)
#       (batch_norm_conv): BatchNormBlock(in_feat: 256, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 256, out_feat: 1024, BN: True, ReLU: False)
#       (unary_shortcut): UnaryBlock(in_feat: 512, out_feat: 1024, BN: True, ReLU: False)
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (13): ResnetBottleneckBlock(
#       (unary1): UnaryBlock(in_feat: 1024, out_feat: 256, BN: True, ReLU: True)
#       (KPConv): KPConv(radius: 0.80, in_feat: 256, out_feat: 256)
#       (batch_norm_conv): BatchNormBlock(in_feat: 256, momentum: 0.050, only_bias: False)
#       (unary2): UnaryBlock(in_feat: 256, out_feat: 1024, BN: True, ReLU: False)
#       (unary_shortcut): Identity()
#       (leaky_relu): LeakyReLU(negative_slope=0.1)
#     )
#     (14): GlobalAverageBlock()
#   )
#   (head_mlp): UnaryBlock(in_feat: 1024, out_feat: 1024, BN: False, ReLU: True)
#   (head_softmax): UnaryBlock(in_feat: 1024, out_feat: 40, BN: False, ReLU: False)
#   (criterion): CrossEntropyLoss()
#   (l1): L1Loss()
# )>