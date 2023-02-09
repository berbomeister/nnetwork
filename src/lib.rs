use anyhow::Ok;
use anyhow::Result;
use tch::nn::{ModuleT, OptimizerConfig, SequentialT};
use tch::{nn, Device};

pub fn learning_rate(epoch: i64) -> f64 {
    if epoch < 15 {
        0.1
    } else if epoch < 30 {
        0.01
    } else {
        0.005
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

pub fn layer<'a>(
    vs: &nn::Path,
    in_channels: i64,
    out_channels: i64,
    kernel_size: i64,
    stride: Option<i64>,
    padding: Option<i64>,
    bias: bool,
    maxpool_kernel: i64,
    dropout: Option<f64>,
) -> nn::FuncT<'a> {
    let pre = conv2d_sublayer(
        &vs.sub("pre"),
        in_channels,
        out_channels / 2,
        kernel_size,
        stride,
        padding,
        bias,
    );
    let post = conv2d_sublayer(
        &vs.sub("pre"),
        out_channels / 2,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
    );
    nn::func_t(move |x, train| {
        let res_ = x.apply_t(&pre, train);
        let res = res_
            .apply_t(&post, train)
            .max_pool2d_default(maxpool_kernel);
        match dropout {
            Some(p) => res.dropout(p, train),
            _ => res,
        }
    })
}

pub fn cnn1(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(layer(
            &vs.sub("layer1"),
            3,
            64,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer2"),
            64,
            128,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer2"),
            128,
            256,
            3,
            Some(1),
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add_fn(|x| x.flatten(1, 3).relu())
        .add(nn::linear(
            &vs.sub("output1"),
            256 * 4 * 4,
            512,
            Default::default(),
        ))
        .add_fn(|x| x.relu())
        .add(nn::linear(&vs.sub("output2"), 512, 10, Default::default()))
}
pub fn fast_resnet2(vs: &nn::Path) -> SequentialT {
    nn::seq_t()
        .add(layer(
            &vs.sub("layer1"),
            3,
            256,
            3,
            None,
            Some(1),
            true,
            4,
            Some(0.25),
        ))
        .add(layer(
            &vs.sub("layer2"),
            256,
            512,
            3,
            None,
            Some(1),
            true,
            2,
            Some(0.25),
        ))
        .add_fn(|x| x.max_pool2d_default(4).flat_view())
        .add(nn::linear(vs.sub("linear"), 512, 10, Default::default()))
        .add_fn(|x| x * 0.125)
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

pub fn test() -> Result<()> {
    let m = tch::vision::cifar::load_dir("data")?;
    let vs = nn::VarStore::new(Device::cuda_if_available());
    let mut user_input = String::new();
    let stdin = std::io::stdin();
    stdin.read_line(&mut user_input)?;
    println!("{:?}", user_input.as_str().trim());
    let net = match user_input.as_str().trim() {
        "1" => fast_resnet(&vs.root()),
        "2" => fast_resnet2(&vs.root()),
        _ => cnn1(&vs.root()),
    };
    let mut opt = nn::Sgd {
        momentum: 0.9,
        dampening: 0.,
        wd: 5e-4,
        nesterov: true,
    }
    .build(&vs, 0.)?;
    for epoch in 1..35 {
        opt.set_lr(learning_rate(epoch));
        for (i, (bimages, blabels)) in m
            .train_iter(64)
            .shuffle()
            .to_device(vs.device())
            .enumerate()
        {
            let bimages = tch::vision::dataset::augmentation(&bimages, true, 4, 8);
            let pred = net.forward_t(&bimages, true);
            let loss = pred.cross_entropy_for_logits(&blabels);
            opt.backward_step(&loss);
            if i % 100 == 0 {
                let test_accuracy =
                    net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
                println!(
                    "epoch: {:4}, batch: {:5}, test acc: {:5.2}%",
                    epoch,
                    i,
                    100. * test_accuracy,
                );
            }
        }
        let test_accuracy =
            net.batch_accuracy_for_logits(&m.test_images, &m.test_labels, vs.device(), 512);
        println!("epoch: {:4} test acc: {:5.2}%", epoch, 100. * test_accuracy,);
    }
    Ok(())
}
