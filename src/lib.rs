use tch::{nn::SequentialT, *};

pub fn learning_rate(epoch: i64) -> f64 {
    if epoch < 50 {
        0.1
    } else if epoch < 100 {
        0.01
    } else {
        0.001
    }
}
pub fn conv2d_sublayer(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
) -> SequentialT {
    let padding = if let Some(val) = padding {
        val
    } else {
        0 //default value
    };
    let stride = if let Some(val) = stride {
        val
    } else {
        1 //default value
    };

    let conv2d_cfg = nn::ConvConfig {
        stride: stride,
        padding: padding,
        bias: bias,
        ..Default::default()
    };
    nn::seq_t()
        .add(nn::conv2d(
            vs,
            in_channels,
            out_channels,
            kernel_size,
            conv2d_cfg,
        ))
        .add_fn(|x| x.relu())
        .add(nn::batch_norm2d(vs, out_channels, Default::default()))
}

pub fn cnn_net(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(conv2d_sublayer(
            &vs.sub("conv1"),
            3,
            64,
            3,
            Some(1),
            Some(1),
            true,
        ))
        .add(conv2d_sublayer(
            &vs.sub("conv2"),
            64,
            64,
            3,
            Some(1),
            Some(1),
            true,
        ))
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn_t(|x, train| x.dropout(0.25, train))
        .add(conv2d_sublayer(
            &vs.sub("conv3"),
            64,
            128,
            3,
            Some(1),
            Some(1),
            true,
        ))
        .add(conv2d_sublayer(
            &vs.sub("conv4"),
            128,
            128,
            3,
            Some(1),
            Some(1),
            true,
        ))
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn_t(|x, train| x.dropout(0.25, train))
        .add_fn(|x| {
            // println!("first{:?}", x.size());
            x.flatten(1,3).relu()
        })
        .add(nn::linear(
            &vs.sub("output1"),
            128 * 8 * 8,
            512,
            Default::default(),
        ))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.sub("output2"), 512, 10, Default::default()))
}
pub fn fast_resnet(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(conv2d_sublayer(
            &vs.sub("pre"),
            3,
            64,
            3,
            None,
            Some(1),
            true,
        ))
        // .add(layer(&vs.sub("layer1"), 64, 128))
        // .add(conv_bn(&vs.sub("inter"), 128, 256))
        .add(conv2d_sublayer(
            &vs.sub("inter"),
            64,
            256,
            3,
            None,
            Some(1),
            true,
        ))
        // .add_fn(|x| x.max_pool2d_default(2))
        // .add_fn(|x| x.max_pool2d_default(2))
        .add_fn(|x| x.max_pool2d_default(4))
        .add_fn_t(|x, train| x.dropout(0.25, train))
        .add(conv2d_sublayer(
            &vs.sub("inter2"),
            256,
            512,
            3,
            None,
            Some(1),
            true,
        ))
        .add_fn(|x| x.max_pool2d_default(2))
        // .add(layer(&vs.sub("layer2"), 256, 512))
        .add_fn(|x| x.max_pool2d_default(4).flat_view())
        .add(nn::linear(vs.sub("linear"), 512, 10, Default::default()))
        .add_fn(|x| x * 0.125)
}
