use crate::file_loader::{FileLoader, GameFilePath};

impl FileLoader {
    pub async fn read(path: &GameFilePath) -> anyhow::Result<Vec<u8>> {
        let path = path.resolve();
        std::fs::read(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
    }

    pub async fn read_to_string(path: &GameFilePath) -> anyhow::Result<String> {
        let path = path.resolve();
        std::fs::read_to_string(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
    }

    pub fn open_file(path: &GameFilePath) -> anyhow::Result<std::fs::File> {
        let path = path.resolve();
        std::fs::File::open(&path).map_err(|err| anyhow::anyhow!("{err} ({})", path.display()))
    }
}
