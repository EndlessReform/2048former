use std::collections::HashMap;

use anyhow::Result;
use parking_lot::Mutex;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct MetaRecord {
    pub seed: u64,
    pub num_moves: usize,
    pub score: u64,
    #[serde(rename = "max_tile")]
    pub highest_tile: u32,
    #[serde(default)]
    pub max_rank: Option<u8>,
}

#[derive(Default)]
pub(crate) struct ValuationEncoder {
    inner: Mutex<ValuationInner>,
}

#[derive(Default)]
struct ValuationInner {
    map: HashMap<String, u8>,
    names: Vec<String>,
}

impl ValuationEncoder {
    pub(crate) fn new() -> Self {
        let mut inner = ValuationInner::default();
        for name in ["search", "tuple11", "tuple10"] {
            inner.register(name);
        }
        Self {
            inner: Mutex::new(inner),
        }
    }

    pub(crate) fn encode(&self, name: &str) -> Result<u8> {
        let mut inner = self.inner.lock();
        if let Some(&id) = inner.map.get(name) {
            return Ok(id);
        }
        let id = inner.register(name);
        Ok(id)
    }

    pub(crate) fn encode_fast(&self, name: Option<&str>) -> Result<u8> {
        match name {
            None | Some("search") => Ok(0),
            Some("tuple11") => Ok(1),
            Some("tuple10") => Ok(2),
            Some(other) => self.encode(other),
        }
    }

    pub(crate) fn as_vec(&self) -> Vec<String> {
        self.inner.lock().names.clone()
    }
}

impl ValuationInner {
    fn register(&mut self, name: &str) -> u8 {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = self.names.len();
        if id >= u8::MAX as usize {
            panic!("valuation enum overflows u8");
        }
        self.names.push(name.to_string());
        self.map.insert(name.to_string(), id as u8);
        id as u8
    }
}
