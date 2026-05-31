用论文"D:\360MoveData\Users\PC\Desktop\iGEM\鸡尾酒酶配比\01_sequences\60 个 cellulosomal components.pdf"找到了网站[CAZy - Bacteria](https://www.cazy.org/b1485.html)
找到了三个酶的NCBI accession并下载



好，我们就按**路线一**整理成一个清晰工作单。你现在只需要做“结构域自动注释”，不用自己读一大堆域图。

路线一的目标是：

> 输入三条完整 FASTA，输出一个表格：每个酶哪一段是信号肽、哪一段是 GH 催化结构域、有没有 CBM、有没有 Dockerin，以及第一轮 MD 应该保留哪一段。

------

# 总流程

```text
三条完整 FASTA
    ↓
InterProScan 网页：识别 GH / Dockerin / CBM 等结构域
    ↓
SignalP 网页：识别信号肽
    ↓
dbCAN 本地版：专门验证 CAZyme 结构域
    ↓
整理成结构域分析总表
    ↓
确定第一轮 MD construct
```

今天先不做 MD。
今天的目标是把三条酶的结构域边界搞清楚。

------

# 0. 建立项目文件夹

在 Ubuntu 里运行：

```bash
mkdir -p ~/cellulase_ratio_model
cd ~/cellulase_ratio_model

mkdir -p 01_sequences
mkdir -p 02_interpro
mkdir -p 03_signalp
mkdir -p 04_dbcan
mkdir -p 05_domain_summary
```

目录含义：

```text
01_sequences       放三条完整 FASTA
02_interpro        放 InterProScan 网页结果
03_signalp         放 SignalP 网页结果
04_dbcan           放 dbCAN 本地结果
05_domain_summary  放最终整理表
```

------

# 1. 准备三条 FASTA

你现在已经找到三个 NCBI accession：

| 酶     | NCBI accession | 长度   |
| ------ | -------------- | ------ |
| Cel5L  | ADU74872.1     | 526 aa |
| Cel9K  | ADU74865.1     | 895 aa |
| Cel48S | ADU75731.1     | 744 aa |

去 NCBI 页面点 **FASTA**，分别复制序列。

然后新建一个合并文件：

```bash
cd ~/cellulase_ratio_model/01_sequences
nano three_cellulases_full.fasta
```

格式整理成这样：

```fasta
>Cel5L_Clo1313_1816_ADU74872.1_full_length_526aa
这里粘贴Cel5L序列

>Cel9K_Clo1313_1809_ADU74865.1_full_length_895aa
这里粘贴Cel9K序列

>Cel48S_Clo1313_2747_ADU75731.1_full_length_744aa
这里粘贴Cel48S序列
```

保存后检查一下：

```bash
grep ">" three_cellulases_full.fasta
```

你应该看到：

```text
>Cel5L_Clo1313_1816_ADU74872.1_full_length_526aa
>Cel9K_Clo1313_1809_ADU74865.1_full_length_895aa
>Cel48S_Clo1313_2747_ADU75731.1_full_length_744aa
```

再检查长度：

```bash
awk '/^>/{if(seq){print name, length(seq)}; name=$0; seq=""; next}{seq=seq$0}END{print name, length(seq)}' three_cellulases_full.fasta
```

理想输出应该接近：

```text
>Cel5L... 526
>Cel9K... 895
>Cel48S... 744
```

如果长度不对，说明复制 FASTA 时漏了或多了东西。

------

# 2. InterProScan 网页注释

这一步用网页，不在本地装。因为本地 InterProScan 很大，不值得为三条序列折腾。

打开：

```text
https://www.ebi.ac.uk/interpro/search/sequence/
```

操作：

1. 上传或粘贴 `three_cellulases_full.fasta`

2. 提交

3. 等结果出来

4. 下载结果，优先下载：

   ```text
   TSV
   ```

5. 保存到：

   ```text
   ~/cellulase_ratio_model/02_interpro/interpro_result.tsv
   ```

你要从 InterPro 里找这些词：

```text
Glycoside hydrolase family 5
Glycoside hydrolase family 9
Glycoside hydrolase family 48
Dockerin
Carbohydrate-binding module
CBM
Cellulose-binding
```

InterProScan 的结果适合识别结构域、功能位点和蛋白家族，它的 TSV 结果里会给出命中的起止坐标，正好对应“哪几位到哪几位是什么”。([Project name not set](https://run-dbcan.readthedocs.io/en/latest/getting_started/installation.html?utm_source=chatgpt.com))

------

# 3. SignalP 网页预测信号肽

打开：

```text
https://services.healthtech.dtu.dk/services/SignalP-6.0/
```

操作：

1. 上传 `three_cellulases_full.fasta`

2. Organism group 选：

   ```text
   Gram-positive
   ```

3. 运行

4. 下载结果表

5. 保存到：

   ```text
   ~/cellulase_ratio_model/03_signalp/signalp_result.txt
   ```

你只关心两件事：

```text
是否有信号肽？
切割位点在哪里？
```

例如如果它写：

```text
CS pos: 28-29
```

那你就记：

```text
信号肽：1-28
成熟蛋白从 29 开始
```

第一轮 MD 一般删除信号肽，因为它是分泌定位用的，不是成熟纤维素酶的功能结构。

------

# 5. 整理 InterPro 结果

等你下载 `interpro_result.tsv` 后，新建脚本：

```bash
cd ~/cellulase_ratio_model/05_domain_summary
nano parse_interpro.py
```

粘贴：

```python
import pandas as pd
from pathlib import Path

infile = Path("../02_interpro/interpro_result.tsv")
outfile = Path("interpro_domain_summary.tsv")

cols = [
    "protein_id",
    "md5",
    "length",
    "analysis",
    "signature_id",
    "signature_desc",
    "start",
    "end",
    "score",
    "status",
    "date",
    "interpro_id",
    "interpro_desc",
    "go",
    "pathway"
]

df = pd.read_csv(infile, sep="\t", header=None)
df = df.iloc[:, :min(len(cols), df.shape[1])]
df.columns = cols[:df.shape[1]]

keywords = [
    "glycoside",
    "hydrolase",
    "Glyco_hydro",
    "GH5",
    "GH9",
    "GH48",
    "dockerin",
    "Dockerin",
    "cellulose",
    "cellulose-binding",
    "carbohydrate-binding",
    "CBM",
    "binding module"
]

pattern = "|".join(keywords)

mask = False
for col in ["signature_desc", "interpro_desc"]:
    if col in df.columns:
        mask = mask | df[col].astype(str).str.contains(pattern, case=False, na=False)

out_cols = [
    c for c in [
        "protein_id",
        "length",
        "analysis",
        "signature_id",
        "signature_desc",
        "start",
        "end",
        "interpro_id",
        "interpro_desc"
    ] if c in df.columns
]

out = df.loc[mask, out_cols].sort_values(["protein_id", "start", "end"])
out.to_csv(outfile, sep="\t", index=False)

print(out.to_string(index=False))
print(f"\nSaved: {outfile}")
```

运行：

```bash
python parse_interpro.py
```

输出文件：

```text
05_domain_summary/interpro_domain_summary.tsv
```

这个文件就是第一版结构域摘要。

------

# 6. 最终你要填的表

新建：

```bash
nano ~/cellulase_ratio_model/05_domain_summary/domain_summary_v0.1.md
```

填这个模板：

```markdown
# 三种纤维素酶结构域自动注释结果 v0.1

| 酶 | NCBI accession | 长度 | 信号肽 SignalP | InterPro 催化结构域 | dbCAN CAZyme 结果 | CBM | Dockerin | 第一轮 MD construct |
|---|---|---:|---|---|---|---|---|---|
| Cel5L | ADU74872.1 | 526 aa | 待填 | GH5：待填 | GH5：待填 | 待填 | 待填 | 待定 |
| Cel9K | ADU74865.1 | 895 aa | 待填 | GH9：待填 | GH9：待填 | 待填 | 待填 | 待定 |
| Cel48S | ADU75731.1 | 744 aa | 待填 | GH48：待填 | GH48：待填 | 待填 | 待填 | 待定 |
```

你后面只要把 InterPro、SignalP、dbCAN 的结果填进去。

------

# 7. 今天的工作顺序

你就按这个顺序来，不要跳。

## 第一步

下载三条 FASTA，合并成：

```text
three_cellulases_full.fasta
```

## 第二步

检查长度：

```bash
awk '/^>/{if(seq){print name, length(seq)}; name=$0; seq=""; next}{seq=seq$0}END{print name, length(seq)}' three_cellulases_full.fasta
```

## 第三步

上传 InterProScan 网页，下载 TSV。

## 第四步

上传 SignalP 6.0 网页，下载结果。

## 第五步

用 micromamba 安装 dbCAN：

```bash
micromamba create -n dbcan -c conda-forge -c bioconda dbcan -y
micromamba activate dbcan
```

## 第六步

跑 dbCAN，得到 `overview.tsv`。

## 第七步

把三个结果整理进 `domain_summary_v0.1.md`。



## [AlphaFold Server](https://alphafoldserver.com/)预测结构



下载后check_cif_lengths_no_install.py检查cif质量

结果：
(base) white@LAPTOP-9J7EHQ7N:~/igem/Cel$ python check_cif_lengths_no_install.py
fold_cel48s_model_0.cif: residues=645, expected=645 -> OK
  mean pLDDT=98.55, CA pLDDT<70=1, CA pLDDT<50=0
fold_cel5l_model_0.cif: residues=428, expected=428 -> OK
  mean pLDDT=95.41, CA pLDDT<70=20, CA pLDDT<50=5
fold_cel9k_model_0.cif: residues=800, expected=800 -> OK
  mean pLDDT=96.49, CA pLDDT<70=13, CA pLDDT<50=11

list_low_plddt_regions.py检查低置信度区域

===========================================================================
fold_cel48s_model_0.cif
Total residues: 645

pLDDT < 70 regions:
  645

pLDDT < 50 regions:

================================================================================
fold_cel5l_model_0.cif
Total residues: 428

pLDDT < 70 regions:
  409-428

pLDDT < 50 regions:
  413-417

================================================================================
fold_cel9k_model_0.cif
Total residues: 800

pLDDT < 70 regions:
  1
  789-800

pLDDT < 50 regions:
  790-800

