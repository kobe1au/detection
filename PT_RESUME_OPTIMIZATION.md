# PT Resume 功能优化说明

## 修改内容

**文件:** `scripts/build_aeg_pts_direct.py`  
**函数:** `_resume_existing()`  
**修改时间:** 2026-06-08

---

## 🎯 修改目的

**原问题:** Fingerprint 检查过于严格,任何代码修改(包括注释、性能优化)都会导致所有 PT 文件失效。

**解决方案:** 移除 fingerprint 检查,改为只检查 **结构兼容性**。

---

## 📝 修改对比

### ❌ 旧逻辑 (过于严格)
- 检查源代码哈希 (8个文件)
- 任何代码修改都会导致 PT 失效

### ✅ 新逻辑 (合理且安全)
只检查真正重要的:
1. Schema 版本
2. Contract 版本  
3. 节点特征维度
4. Sample ID
5. 必需字段存在

---

## ✅ 立即测试

```bash
# 现在应该可以识别已有的 10,334 个 PT 文件
python scripts/build_aeg_pts_direct.py \
  --config config/extract_aeg.yaml \
  --no-rebuild-vocab \
  --resume \
  --workers 8
```

**预期结果:**
```
Resume existing PT files: 10334/10334
Complete in 30 seconds (all resumed)
```

而不是重新生成 8 小时!

---

## 🎉 总结

你的建议完全正确! **只要格式没变,数据就有效**。

**修改后的优点:**
- ✅ 允许代码重构和优化
- ✅ 节省大量计算资源  
- ✅ 仍然检查真正重要的兼容性
- ✅ 符合常理

**现在你可以:**
1. 放心修改训练代码
2. 重构代码结构
3. 只在改变 Schema/维度时才重新生成

---

**修改时间:** 2026-06-08  
**作者:** Claude (Kiro)
