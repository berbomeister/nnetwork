use std::borrow::Borrow;
use tch::*;
#[derive(Debug)]
pub struct MultiHeadAttentionLayer {
    pub w_q: nn::Linear,
    pub w_k: nn::Linear,
    pub w_v: nn::Linear,
    pub n_heads: i64,
    pub hid_size: i64,
    pub scale: Tensor,
}

pub fn mhal<'a, T: Borrow<nn::Path<'a>>>(
    vs: T,
    hid_size: i64,
    n_heads: i64,
) -> MultiHeadAttentionLayer {
    let vs = vs.borrow();
    let value: Tensor = Tensor::from(hid_size as f64).set_requires_grad(false);
    MultiHeadAttentionLayer {
        w_q: nn::linear(vs / "query", hid_size, hid_size, Default::default()),
        w_k: nn::linear(vs / "keys", hid_size, hid_size, Default::default()),
        w_v: nn::linear(vs / "values", hid_size, hid_size, Default::default()),
        n_heads: n_heads,
        hid_size: hid_size,
        scale: value.sqrt(),
    }
}
impl MultiHeadAttentionLayer {
    pub fn forward_t(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
        train: bool
    ) -> Tensor {
        let batch_size = queries.size()[0];
        let q = queries.apply(&self.w_q);
        let k = keys.apply(&self.w_k);
        let v = values.apply(&self.w_v);
        let q = q.view((batch_size,-1,self.n_heads,self.hid_size / self.n_heads)).transpose(1, 2);
        let k = k.view((batch_size,-1,self.n_heads,self.hid_size / self.n_heads)).transpose(1, 2).transpose(2, 3);
        let v = v.view((batch_size,-1,self.n_heads,self.hid_size / self.n_heads)).transpose(1, 2);
        let q : Tensor = q / self.scale;
        let scores = if let Some(mask) = mask {
            q.matmul(&k) + mask
        } else {
            q.matmul(&k)
        };
        let attention = scores
        .softmax(-1, scores.kind());
        let x = attention.matmul(&v);
        let x = x.transpose(1, 2).flatten(2,3);
        x
    }
}
