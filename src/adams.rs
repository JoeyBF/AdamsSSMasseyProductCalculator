mod element;
mod generator;
mod masseyproduct;
mod multiplication;

use std::cmp::{Ordering, PartialOrd};

use std::fmt;
use std::fmt::{Display, Formatter};

use std::ops::{Add, AddAssign};

use crate::lattice::{join, meet, JoinSemilattice, MeetSemilattice};

pub use element::AdamsElement;
pub use generator::AdamsGenerator;
pub use masseyproduct::MasseyProduct;
pub use multiplication::AdamsMultiplication;

/// type synonym for (s,t) bidegrees

#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bidegree {
    /// resolution degree
    s: u32,
    /// internal degree
    t: i32,
}

/// iterates over a rectangle of (s,t) degrees specified by min and max inclusive
/// iterates over t first (i.e. if this were a pair of loops, s is the outer loop, t is the inner loop)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BidegreeIterator {
    /// Bidegree to start at
    min: Bidegree,
    /// Bidegree to iterate over (includes this degree)
    max: Bidegree,
    /// current location in iteration
    current: Bidegree,
}

impl BidegreeIterator {
    pub fn new(min: Bidegree, max: Bidegree) -> Self {
        BidegreeIterator {
            min,
            max,
            current: min,
        }
    }
    pub fn new_from_origin(max: Bidegree) -> Self {
        Self::new((0, 0).into(), max)
    }
}

impl From<Bidegree> for BidegreeIterator {
    fn from(deg: Bidegree) -> Self {
        Self::new((0, 0).into(), deg)
    }
}
impl<'a> From<&'a Bidegree> for BidegreeIterator {
    fn from(deg: &'a Bidegree) -> Self {
        Self::new((0, 0).into(), *deg)
    }
}

impl Iterator for BidegreeIterator {
    type Item = Bidegree;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.max {
            if self.current.t < self.max.t {
                self.current.t += 1;
            } else {
                self.current.t = self.min.t;
                self.current.s += 1;
            }
            Some(self.current)
        } else {
            None
        }
    }
}

/// iterates over a rectangle of (n,s) degrees specified by min and max inclusive
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StemIterator {
    /// Bidegree to start at
    min: Bidegree,
    /// Bidegree to iterate over (includes this degree)
    max: Bidegree,
    /// current location in iteration
    current: Bidegree,
}

// might want to split Bidegrees and StemDegrees, because they should have different ordering properties
impl StemIterator {
    pub fn new(min: Bidegree, max: Bidegree) -> Self {
        StemIterator {
            min,
            max,
            current: min,
        }
    }
    pub fn new_from_origin(max: Bidegree) -> Self {
        Self::new((0, 0).into(), max)
    }
}

impl From<Bidegree> for StemIterator {
    fn from(deg: Bidegree) -> Self {
        Self::new((0, 0).into(), deg)
    }
}
impl<'a> From<&'a Bidegree> for StemIterator {
    fn from(deg: &'a Bidegree) -> Self {
        Self::new((0, 0).into(), *deg)
    }
}

impl Iterator for StemIterator {
    type Item = Bidegree;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.n() <= self.max.n()
            && self.current.s() <= self.max.s()
            && self.current != self.max
        {
            if self.current.n() < self.max.n() {
                self.current.t += 1; // n = t-s, so increment t to increment n
            } else {
                self.current.s += 1; // increment s first
                self.current.t = self.min.n() + self.current.s as i32; // sets n to self.min.n()
            }
            Some(self.current)
        } else {
            None
        }
    }
}

impl Bidegree {
    pub fn s(&self) -> u32 {
        self.s
    }
    pub fn s_mut(&mut self) -> &mut u32 {
        &mut self.s
    }
    pub fn t(&self) -> i32 {
        self.t
    }
    pub fn t_mut(&mut self) -> &mut i32 {
        &mut self.t
    }
    pub fn n(&self) -> i32 {
        self.t - self.s as i32
    }
    pub fn new(s: u32, t: i32) -> Self {
        Self { s, t }
    }

    pub fn iter_stem(&self) -> StemIterator {
        StemIterator::from(self)
    }

    pub fn iter_s_t(&self) -> BidegreeIterator {
        BidegreeIterator::from(self)
    }

    /// Checks that the difference in s degrees is nonnegative.
    /// Returns difference as a bidegree if so, otherwise returns None.
    pub fn try_subtract<T: Into<Bidegree>>(&self, smaller: T) -> Option<Bidegree> {
        let smaller = smaller.into();
        if self.s >= smaller.s {
            Some(Bidegree {
                s: self.s - smaller.s,
                t: self.t - smaller.t,
            })
        } else {
            None
        }
    }

    /// Computes the bidegree containing the Massey product of elements in the given bidegrees.
    pub fn massey_bidegree(a: Bidegree, b: Bidegree, c: Bidegree) -> Bidegree {
        (a + b + c)
            .try_subtract((1, 0))
            .expect("Trying to compute <1,1,1>")
    }
}

impl PartialOrd for Bidegree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let (s1, t1): (u32, i32) = self.into();
        let (s2, t2) = other.into();
        if s1 == s2 && t1 == t2 {
            Some(Ordering::Equal)
        } else if s1 <= s2 && t1 <= t2 {
            Some(Ordering::Less)
        } else if s1 >= s2 && t1 >= t2 {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

impl Display for Bidegree {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "({}, {})", self.n(), self.s())
    }
}

impl Add for Bidegree {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            s: self.s + other.s,
            t: self.t + other.t,
        }
    }
}
impl AddAssign for Bidegree {
    fn add_assign(&mut self, other: Self) {
        self.s += other.s;
        self.t += other.t;
    }
}

impl From<(u32, i32)> for Bidegree {
    fn from(tuple: (u32, i32)) -> Self {
        Self::new(tuple.0, tuple.1)
    }
}

impl From<Bidegree> for (u32, i32) {
    fn from(deg: Bidegree) -> Self {
        (deg.s(), deg.t())
    }
}

impl<'a> From<&'a Bidegree> for (u32, i32) {
    fn from(deg: &'a Bidegree) -> Self {
        (deg.s(), deg.t())
    }
}

impl<'a> From<&'a Bidegree> for (&'a u32, &'a i32) {
    fn from(deg: &'a Bidegree) -> Self {
        (&deg.s, &deg.t)
    }
}

/*
impl Save for Bidegree {
    fn save(&self, buffer: &mut impl Write) -> io::Result<()> {
        self.s.save(buffer)?;
        self.t.save(buffer)?;
        Ok(())
    }
}

impl Load for Bidegree {
    type AuxData = ();

    fn load(buffer: &mut impl Read, _: &Self::AuxData) -> io::Result<Self> {
        let s = u32::load(buffer, &())?;
        let t = i32::load(buffer, &())?;
        Ok(Bidegree { s, t })
    }
}
*/

impl MeetSemilattice for Bidegree {
    fn meet(self, rhs: Bidegree) -> Bidegree {
        Bidegree {
            s: meet(self.s, rhs.s),
            t: meet(self.t, rhs.t),
        }
    }
}
impl JoinSemilattice for Bidegree {
    fn join(self, rhs: Bidegree) -> Bidegree {
        Bidegree {
            s: join(self.s, rhs.s),
            t: join(self.t, rhs.t),
        }
    }
}
