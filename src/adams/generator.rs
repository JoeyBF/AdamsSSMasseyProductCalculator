use super::{AdamsElement, Bidegree};

use std::fmt;
use std::fmt::{Display, Formatter};

//type AdamsGenerator = (u32, i32, usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AdamsGenerator {
    /// resolution degree
    s: u32,
    /// internal degree
    t: i32,
    /// generator index
    idx: usize,
}

impl AdamsGenerator {
    pub fn new(s: u32, t: i32, idx: usize) -> AdamsGenerator {
        AdamsGenerator { s, t, idx }
    }

    pub fn s(&self) -> u32 {
        self.s
    }

    pub fn t(&self) -> i32 {
        self.t
    }

    pub fn degree(&self) -> Bidegree {
        (self.s, self.t).into()
    }

    pub fn n(&self) -> i32 {
        self.t - self.s as i32
    }

    pub fn idx(&self) -> usize {
        self.idx
    }

    pub fn vector(&self, dim: usize) -> Vec<u32> {
        let mut ret = vec![0; dim];
        ret[self.idx] = 1;
        ret
    }
}

impl Display for AdamsGenerator {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {}, {})", self.n(), self.s(), self.idx())
    }
}

impl From<(u32, i32, usize)> for AdamsGenerator {
    fn from(tuple: (u32, i32, usize)) -> Self {
        Self::new(tuple.0, tuple.1, tuple.2)
    }
}
impl From<(Bidegree, usize)> for AdamsGenerator {
    fn from(tuple: (Bidegree, usize)) -> Self {
        let (deg, idx) = tuple;
        let (s, t) = deg.into();
        Self::new(s, t, idx)
    }
}

impl From<AdamsGenerator> for (u32, i32, usize) {
    fn from(gen: AdamsGenerator) -> Self {
        (gen.s(), gen.t(), gen.idx())
    }
}

impl TryFrom<AdamsElement> for AdamsGenerator {
    type Error = ();

    fn try_from(value: AdamsElement) -> Result<Self, Self::Error> {
        let (s, t, v) = value.into();
        if v.iter().sum::<u32>() == 1 {
            let (idx, _) = v.first_nonzero().unwrap();
            Ok((s, t, idx).into())
        } else {
            Err(())
        }
    }
}

impl<'a> TryFrom<&'a AdamsElement> for AdamsGenerator {
    type Error = ();

    fn try_from(value: &'a AdamsElement) -> Result<Self, Self::Error> {
        let (s, t, v) = value.into();
        if v.iter().sum::<u32>() == 1 {
            let (idx, _) = v.first_nonzero().unwrap();
            Ok((s, t, idx).into())
        } else {
            Err(())
        }
    }
}
