use std::path::{Path, PathBuf};

lazy_static::lazy_static! {
    pub static ref IKARI_PATH_MAKER: GamePathMaker = GamePathMaker::new(Some(PathBuf::from("ikari")));
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct GameFilePath {
    root: PathBuf,
    pub relative_path: PathBuf,
}

impl GameFilePath {
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn resolve(&self) -> PathBuf {
        self.root.join(&self.relative_path)
    }
}

pub struct GamePathMaker {
    root: PathBuf,
}

impl GamePathMaker {
    pub fn new(root: Option<PathBuf>) -> Self {
        Self {
            root: root.unwrap_or_else(|| "".into()),
        }
    }

    pub fn make<T>(&self, relative_path: T) -> GameFilePath
    where
        T: Into<PathBuf>,
    {
        GameFilePath {
            root: self.root.clone(),
            relative_path: relative_path.into(),
        }
    }
}

pub struct FileLoader;
