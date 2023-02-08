#![allow(dead_code)]
const LABELS: i64 = 10; // number of distinct labels
const HEIGHT: usize = 28; 
const WIDTH: usize = 28;
const IMAGE_DIM: i64 = 784;
const HIDDEN_NODES: i64 = 128;

const TRAIN_SIZE: usize = 50000;
const VAL_SIZE: usize = 10000;
const TEST_SIZE: usize =10000;

const N_EPOCHS: i64 = 200;

const THRES: f64 = 0.001;

const BATCH_SIZE: i64 = 256;