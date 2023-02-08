#![allow(dead_code)]
use tch::{nn::SequentialT, *};
fn conv2d_sublayer(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
) -> nn::SequentialT {
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
            vs / "conv2d",
            in_channels,
            out_channels,
            kernel_size,
            conv2d_cfg,
        ))
        .add_fn(|x| x.relu())
        .add(nn::batch_norm2d(
            vs / "norm2d",
            out_channels,
            Default::default(),
        ))
}
fn regularization_layer<'a>(maxpool_kernel: i64, p: f64, dropout: bool) -> nn::FuncT<'a> {
    nn::func_t(move |xs, train| {
        let maxpool = xs.max_pool2d_default(maxpool_kernel);
        if dropout {
            let dropout = maxpool.dropout(p, train);
            println!("{:?}, {:?}",dropout.size(),maxpool.size());
            return maxpool + dropout;
        }
        maxpool
    })
}
fn output_layer(
    vs: &nn::Path,
    in_channels: i64,
    hid_channels: i64,
    out_channels: i64,
    maxpool_kernel: i64,
) -> nn::SequentialT {
    nn::seq_t()
        .add(nn::linear(
            vs / "linear1",
            in_channels,
            hid_channels,
            Default::default(),
        ))
        .add_fn(|x| {println!("{:?}",x.size());x.relu()})
        .add_fn(move |x| x.max_pool2d_default(maxpool_kernel).flat_view())
        .add(nn::linear(
            vs / "linear2",
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
            None,
            Some(1),
            true,
        ))
        .add(conv2d_sublayer(
            &vs.sub("conv2"),
            64,
            64,
            3,
            None,
            Some(1),
            true,
        ))
        .add(regularization_layer(2, 0.25 as f64, true))
        .add(conv2d_sublayer(
            &vs.sub("conv3"),
            64,
            128,
            3,
            None,
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
        .add(regularization_layer(2, 0.25 as f64, true))
        .add(output_layer(&vs.sub("output"), 128, 512, 10, 2))
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