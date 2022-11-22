use std::cmp::Ordering;

/// PartialOrder with greatest lower bounds, or meets
pub trait MeetSemilattice: PartialOrd<Self> {
    fn meet(self, rhs: Self) -> Self;
}

pub fn meet<T>(v1: T, v2: T) -> T
where
    T: MeetSemilattice,
{
    T::meet(v1, v2)
}

/// Partial Order with least upper bounds, or joins
pub trait JoinSemilattice: PartialOrd<Self> {
    fn join(self, rhs: Self) -> Self;
}

pub fn join<T>(v1: T, v2: T) -> T
where
    T: JoinSemilattice,
{
    T::join(v1, v2)
}

/// A lattice is simultaneously a MeetSemilattice and a JoinSemilattice
pub trait Lattice: MeetSemilattice + JoinSemilattice {}

impl<T: MeetSemilattice + JoinSemilattice> Lattice for T {}

// marker trait to enable automatic Meet and Join Semilattice implementations for things that implement Ord
pub trait LatticeFromOrd: Ord {}

impl LatticeFromOrd for u8 {}
impl LatticeFromOrd for i8 {}
impl LatticeFromOrd for u16 {}
impl LatticeFromOrd for i16 {}
impl LatticeFromOrd for u32 {}
impl LatticeFromOrd for i32 {}
impl LatticeFromOrd for u64 {}
impl LatticeFromOrd for i64 {}
impl LatticeFromOrd for u128 {}
impl LatticeFromOrd for i128 {}
impl LatticeFromOrd for usize {}
impl LatticeFromOrd for isize {}
impl LatticeFromOrd for bool {}

// Too general for trait ord. Use a marker trait
impl<T: LatticeFromOrd> MeetSemilattice for T {
    fn meet(self, rhs: Self) -> Self {
        Self::min(self, rhs)
    }
}
impl<T: LatticeFromOrd> JoinSemilattice for T {
    fn join(self, rhs: Self) -> Self {
        Self::max(self, rhs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WithMin<T> {
    From(T),
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WithMax<T> {
    Max,
    From(T),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WithMinMax<T> {
    Max,
    From(T),
    Min,
}

impl<T> From<Option<T>> for WithMin<T> {
    fn from(val: Option<T>) -> WithMin<T> {
        match val {
            Some(t) => WithMin::From(t),
            None => WithMin::Min,
        }
    }
}
impl<T> From<Option<T>> for WithMax<T> {
    fn from(val: Option<T>) -> WithMax<T> {
        match val {
            Some(t) => WithMax::From(t),
            None => WithMax::Max,
        }
    }
}
impl<T> From<WithMin<T>> for Option<T> {
    fn from(val: WithMin<T>) -> Option<T> {
        match val {
            WithMin::From(t) => Some(t),
            WithMin::Min => None,
        }
    }
}
impl<T> From<WithMax<T>> for Option<T> {
    fn from(val: WithMax<T>) -> Option<T> {
        match val {
            WithMax::From(t) => Some(t),
            WithMax::Max => None,
        }
    }
}

impl<T> From<WithMin<WithMax<T>>> for WithMinMax<T> {
    fn from(val: WithMin<WithMax<T>>) -> WithMinMax<T> {
        match val {
            WithMin::From(wm) => match wm {
                WithMax::From(t) => WithMinMax::From(t),
                WithMax::Max => WithMinMax::Max,
            },
            WithMin::Min => WithMinMax::Min,
        }
    }
}
impl<T> From<WithMax<WithMin<T>>> for WithMinMax<T> {
    fn from(val: WithMax<WithMin<T>>) -> WithMinMax<T> {
        match val {
            WithMax::From(wm) => match wm {
                WithMin::From(t) => WithMinMax::From(t),
                WithMin::Min => WithMinMax::Min,
            },
            WithMax::Max => WithMinMax::Max,
        }
    }
}
impl<T> From<WithMinMax<T>> for Option<T> {
    fn from(val: WithMinMax<T>) -> Option<T> {
        match val {
            WithMinMax::From(t) => Some(t),
            WithMinMax::Min => None,
            WithMinMax::Max => None,
        }
    }
}

impl<T: PartialOrd> PartialOrd for WithMin<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use self::WithMin::*;
        match self {
            From(t1) => match other {
                From(t2) => t1.partial_cmp(t2),
                Min => Some(Ordering::Greater),
            },
            Min => match other {
                From(_) => Some(Ordering::Less),
                Min => Some(Ordering::Equal),
            },
        }
    }
}

impl<T: PartialOrd> PartialOrd for WithMax<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use self::WithMax::*;
        match self {
            Max => match other {
                Max => Some(Ordering::Equal),
                From(_) => Some(Ordering::Greater),
            },
            From(t1) => match other {
                Max => Some(Ordering::Less),
                From(t2) => t1.partial_cmp(t2),
            },
        }
    }
}
impl<T: PartialOrd> PartialOrd for WithMinMax<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        use self::WithMinMax::*;
        match self {
            Max => match other {
                Max => Some(Ordering::Equal),
                From(_) => Some(Ordering::Greater),
                Min => Some(Ordering::Greater),
            },
            From(t1) => match other {
                Max => Some(Ordering::Less),
                From(t2) => t1.partial_cmp(t2),
                Min => Some(Ordering::Greater),
            },
            Min => match other {
                Max => Some(Ordering::Less),
                From(_) => Some(Ordering::Less),
                Min => Some(Ordering::Equal),
            },
        }
    }
}

impl<T: Ord> Ord for WithMin<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        use self::WithMin::*;
        match self {
            From(t1) => match other {
                From(t2) => t1.cmp(t2),
                Min => Ordering::Greater,
            },
            Min => match other {
                From(_) => Ordering::Less,
                Min => Ordering::Equal,
            },
        }
    }
}

impl<T: Ord> Ord for WithMax<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        use self::WithMax::*;
        match self {
            Max => match other {
                Max => Ordering::Equal,
                From(_) => Ordering::Greater,
            },
            From(t1) => match other {
                Max => Ordering::Less,
                From(t2) => t1.cmp(t2),
            },
        }
    }
}
impl<T: Ord> Ord for WithMinMax<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        use self::WithMinMax::*;
        match self {
            Max => match other {
                Max => Ordering::Equal,
                From(_) => Ordering::Greater,
                Min => Ordering::Greater,
            },
            From(t1) => match other {
                Max => Ordering::Less,
                From(t2) => t1.cmp(t2),
                Min => Ordering::Greater,
            },
            Min => match other {
                Max => Ordering::Less,
                From(_) => Ordering::Less,
                Min => Ordering::Equal,
            },
        }
    }
}

impl<T: MeetSemilattice> MeetSemilattice for WithMin<T> {
    fn meet(self, other: Self) -> Self {
        use self::WithMin::*;
        match self {
            From(t1) => match other {
                From(t2) => From(t1.meet(t2)),
                Min => Min,
            },
            Min => Min,
        }
    }
}

impl<T: MeetSemilattice> MeetSemilattice for WithMax<T> {
    fn meet(self, other: Self) -> Self {
        use self::WithMax::*;
        match self {
            Max => other,
            From(t1) => match other {
                Max => From(t1),
                From(t2) => From(t1.meet(t2)),
            },
        }
    }
}

impl<T: MeetSemilattice> MeetSemilattice for WithMinMax<T> {
    fn meet(self, other: Self) -> Self {
        use self::WithMinMax::*;
        match self {
            Max => other,
            From(t1) => match other {
                Max => From(t1),
                From(t2) => From(t1.meet(t2)),
                Min => Min,
            },
            Min => Min,
        }
    }
}

impl<T: JoinSemilattice> JoinSemilattice for WithMin<T> {
    fn join(self, other: Self) -> Self {
        use self::WithMin::*;
        match self {
            From(t1) => match other {
                From(t2) => From(t1.join(t2)),
                Min => From(t1),
            },
            Min => other,
        }
    }
}

impl<T: JoinSemilattice> JoinSemilattice for WithMax<T> {
    fn join(self, other: Self) -> Self {
        use self::WithMax::*;
        match self {
            Max => Max,
            From(t1) => match other {
                Max => Max,
                From(t2) => From(t1.join(t2)),
            },
        }
    }
}

impl<T: JoinSemilattice> JoinSemilattice for WithMinMax<T> {
    fn join(self, other: Self) -> Self {
        use self::WithMinMax::*;
        match self {
            Max => Max,
            From(t1) => match other {
                Max => Max,
                From(t2) => From(t1.join(t2)),
                Min => From(t1),
            },
            Min => other,
        }
    }
}
