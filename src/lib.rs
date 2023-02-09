use tch::{
    nn::{FuncT, SequentialT},
    *,
};

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
fn conv2d_layer<'a>(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
    maxpool_kernel: i64,
    p: Option<f64>,
) -> FuncT<'a> {
    let sublayer = conv2d_sublayer(
        &vs.sub("conv2d"),
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    );
    nn::func_t(move |x, train| {
        let pre = x
            .apply_t(&sublayer, train)
            .max_pool2d_default(maxpool_kernel);
        if let Some(prob) = p {
            pre.dropout(prob, train)
        } else {
            pre
        }
    })
}
fn regularization_layer<'a>(maxpool_kernel: i64, p: f64, dropout: bool) -> nn::FuncT<'a> {
    nn::func_t(move |xs, train| {
        let maxpool = xs.max_pool2d_default(maxpool_kernel);
        // if dropout {
        //     let dropout = maxpool.dropout(p, train);
        //     // println!("{:?}, {:?}", dropout.size(), maxpool.size());
        //     // return maxpool + dropout;
        //     return dropout;
        // }
        maxpool
    })
}
fn output_layer(
    vs: &nn::Path,
    in_channels: i64,
    hid_channels: i64,
    out_channels: i64,
) -> nn::SequentialT {
    nn::seq_t()
        .add_fn(|x| x.flat_view())
        .add(nn::linear(
            vs,
            in_channels,
            hid_channels,
            Default::default(),
        ))
        // .add_fn(|x| x.relu())
        .add(nn::linear(
            vs,
            hid_channels,
            out_channels,
            Default::default(),
        ))
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
        // dropout ?
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
        // dropout ?
        .add(output_layer(&vs.sub("output"), 128 * 8 * 8, 512, 10))
}
pub fn learning_rate(epoch: i64) -> f64 {
    if epoch < 50 {
        0.1
    } else if epoch < 100 {
        0.01
    } else {
        0.001
    }
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
        .add_fn(|x| x.max_pool2d_default(2))
        .add_fn(|x| x.max_pool2d_default(2))
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

