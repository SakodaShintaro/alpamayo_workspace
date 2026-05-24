# alpamayo_workspace

- <https://github.com/NVlabs/alpamayo>
- <https://github.com/NVlabs/alpamayo1.5>
- <https://github.com/NVlabs/alpasim>

をsubmoduleに持ち、環境を1つで管理することを目指すリポジトリ

## Clone

submoduleも一緒に展開するには、以下のいずれか。

### cloneと同時に展開

```bash
git clone --recurse-submodules https://github.com/SakodaShintaro/alpamayo_workspace
```

### 普通にcloneした後に展開

```bash
git clone https://github.com/SakodaShintaro/alpamayo_workspace
cd alpamayo_workspace
git submodule update --init --recursive
```
