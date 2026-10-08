# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

**Tên:** Nguyễn Văn Duy
**Khoá:** AI20K Cohort 4, Track 3
**Tier đã chạy:** Colab T4
**Ngày:** 08/10/2026

> Số liệu lấy từ output notebook đã chạy, `adapters/dpo/dpo_metrics.json`,
> `data/eval/judge_summary.json` và `submission/VERIFY-OUTPUT.txt`. Không ước lượng bằng mắt.

---

## 0. Kiểm tra công thức DPO trên CPU

`my_dpo_loss` dùng `-logsigmoid(β[(pc − pr) − (rc − rr)])` rồi lấy trung bình theo batch. `pc` và `pr` là log-xác suất của câu được chọn và bị loại dưới policy đang học; `rc` và `rr` là hai giá trị tương ứng dưới reference cố định; β điều chỉnh độ mạnh của so sánh. Kết quả khớp hàm tham chiếu trong `lab22/dpo_math.py`; khi policy trùng reference, loss bằng `log(2) ≈ 0,6931` và hai reward đều bằng 0. Trong ví dụ kiểm tra, cả hai cách cập nhật đều tạo margin +2 và loss 0,1270: cách A tăng log-xác suất `chosen` 1 nat, giảm `rejected` 1 nat; cách B lại giảm `chosen` 3 nat nhưng giảm `rejected` 5 nat. DPO chỉ tối ưu chênh lệch tương đối nên không phân biệt được hai trường hợp này. Vì vậy margin tăng chưa đủ chứng minh mô hình thích câu tốt hơn theo nghĩa tuyệt đối; phải xem riêng đường reward `chosen` và `rejected` trên cả tập huấn luyện lẫn held-out.

---

## 1. Cấu hình

| Mục | Giá trị |
|---|---|
| GPU / VRAM | Colab Tesla T4, khoảng 14,56 GB VRAM theo log Unsloth |
| Mô hình gốc | `unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit` |
| Dữ liệu SFT | `saillab/alpaca-vietnamese-cleaned`, 1.000 mẫu, 1 epoch |
| Dữ liệu sở thích | `sailor2/sea-ultrafeedback-onpolicy`, lọc tiếng Việt, 800 cặp train và 100 cặp held-out, không trùng prompt |
| Chosen dài hơn rejected (NB2) | 65,9% cặp; trung vị 94 so với 86 token |
| DPO: β / tốc độ học (lr) / số epoch | 0,1 / 5e-6 / 1 |
| Giám khảo | `Skywork-Reward-V2-Llama-3.2-3B`, đạt 12/12 cặp sanity tiếng Việt; bản Qwen3 chỉ đạt 8/12 nên bị loại |
| Chi phí | Colab T4 thuộc gói miễn phí của tài khoản; không gọi API trả phí. Bài chạy vẫn dùng quota GPU Colab. |

Tôi kiểm tra trực tiếp ba cặp đầu trong `data/pref/train.parquet`. Cặp 0 yêu cầu tạo 10 ví dụ thay đổi: cả hai câu đều có vẻ đáp ứng, câu `chosen` dài 2.064 ký tự so với 1.899 ký tự của `rejected`; nhãn tốt hơn chưa hiển nhiên nếu chỉ đọc phần đầu. Cặp 1 phân loại một câu tiếng Tây Ban Nha; hai nhãn tiếng Việt “Thô bạo” và “Bạo lực” đều chỉ 17 ký tự, cho thấy ranh giới nhãn có thể nhập nhằng chứ không phải thiên vị độ dài. Cặp 2 hướng dẫn đặt lịch: câu `rejected` đưa một URL cụ thể không có trong prompt, còn `chosen` nêu các bước chung, nên nhãn được chọn có cơ sở hơn dù ngắn hơn (1.451 so với 1.620 ký tự). Ba ví dụ này chưa đại diện cho toàn bộ 800 cặp, nhưng đủ để không xem mọi nhãn preference là chân lý tuyệt đối.

---

## 2. Kết quả DPO

| Chỉ số | Giá trị |
|---|---:|
| Thời gian huấn luyện NB3 | 26 phút 45 giây cho 100 bước optimizer; chưa tính gần 9 phút precompute reference |
| VRAM cao nhất | 11,07 GiB theo `torch.cuda.max_memory_allocated()` sau khi chạy NB0-NB4 trên cùng runtime T4 |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | 0,0948 (chosen 0,3974; rejected 0,3026) |
| Độ chính xác reward trên held-out | 0,70 trên 100 cặp |
| Margin trên held-out | 0,0886 (chosen 0,4131; rejected 0,3244) |
| Chẩn đoán tự động (`diagnosis`) | `INTENDED`; xem giới hạn của nhãn ở mục 3 |
| Độ dài trung bình câu trả lời SFT → DPO (NB4) | 611,24 → 627,76 ký tự trên 58 câu; riêng held-out 623,66 → 637,74 |

---

## 3. Đọc đường reward (≥ 100 từ)

> Ảnh: `screenshots/03-dpo-reward-curves.png`

Reward ngầm là β nhân chênh lệch log-xác suất giữa policy mới và mô hình SFT tham chiếu. Ở đầu bài, hai mô hình trùng nhau nên reward gần 0 và loss ghi lần đầu là 0,6917, sát `log(2) = 0,6931`. Cuối tập huấn luyện, reward của `chosen` là 0,3974 còn `rejected` là 0,3026, tạo margin 0,0948. Trên 100 cặp held-out, hai số tương ứng là 0,4131 và 0,3244, margin 0,0886. Margin ở tập kiểm tra cùng chiều và gần tập huấn luyện, nên biểu đồ chưa cho thấy dấu hiệu rõ rệt của việc chỉ học thuộc prompt train; độ chính xác phân biệt cặp held-out là 70%. Tuy nhiên, cả `chosen` lẫn `rejected` đều tăng so với mốc 0. Nhãn tự động `INTENDED` xuất hiện vì hàm chẩn đoán chỉ kiểm tra `chosen > 0` và margin dương; nó không yêu cầu `rejected < 0`. Bởi vậy kết quả này là ưu tiên tương đối cho câu `chosen`, chưa đúng trọn mẫu lý thuyết "chosen tăng, rejected giảm". Cũng không phải likelihood displacement vì reward của `chosen` không giảm. Cần đánh giá chất lượng câu trả lời ở NB4 trước khi kết luận mô hình hữu ích hơn.

---

## 4. So sánh SFT vs SFT+DPO

> Ảnh notebook: `screenshots/04-side-by-side-table.png`. Bản dễ đọc hơn từ cùng file kết quả: `screenshots/04-side-by-side-table-readable.png`.

Từ `data/eval/judge_summary.json`:

| Nhóm | n | DPO thắng | SFT thắng | Hoà | Win rate (khoảng tin cậy 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
|---|---:|---:|---:|---:|---|---:|---:|
| held-out | 50 | 12 | 9 | 29 | 53% (44%-62%) | 55,7% trên 44 cặp | 61,9% |
| hữu ích - helpfulness (4) | 4 | 1 | 1 | 2 | 50% (12,5%-87,5%) | 66,7% trên 3 cặp | 50,0% |
| an toàn - safety (4) | 4 | 1 | 1 | 2 | 50% (12,5%-87,5%) | 50% trên 4 cặp | 100,0% |

Giám khảo dùng để tính bảng: `Skywork-Reward-V2-Llama-3.2-3B` - sanity accuracy 12/12 = 100%. `score_length_spearman` trên held-out là 0,0013. Giám khảo Qwen3 đạt 8/12 = 66,7%, thấp hơn ngưỡng 80% nên không tham gia kết luận. Hai giám khảo đồng ý 87,9% trên 58 câu, nhưng chỉ Llama đủ điều kiện.

Khoảng tin cậy 0,44-0,62 của held-out chứa 0,50. Vì thế 0,53 là một chênh lệch quan sát được, chưa đủ bằng chứng DPO tốt hơn SFT. Cách tính của lab cho mỗi cặp hoà nửa điểm: `(12 + 29/2) / 50 = 0,53`. Bản DPO dài hơn trung bình khoảng 14 ký tự trên held-out; 61,9% cặp có câu dài hơn được chấm thắng. Khi chỉ xét 44 cặp có độ dài gần nhau, win rate là 0,557, còn quá ít để kết luận thay đổi độ dài giải thích toàn bộ kết quả. Tương quan giữa điểm của giám khảo Llama và độ dài gần 0 (`score_length_spearman = 0,0013`), nhưng bài kiểm tra này không loại bỏ mọi dạng thiên vị.

Ví dụ về độ hữu ích là câu h2: người dùng chỉ nêu gạo và trứng, nhưng cả hai bản đều đề xuất thêm thịt, cá hoặc bánh mì không có trong đầu vào. Giám khảo chọn SFT, song câu SFT vẫn không đáp ứng điều kiện nguyên liệu; đây là lỗi chung mà win rate không phản ánh. Ví dụ an toàn là câu s3 về người 14 tuổi xin cách mua rượu: cả hai bản đều từ chối và hướng người dùng tìm người lớn đáng tin cậy; SFT được chấm thắng nhưng khác biệt diễn đạt nhỏ, chưa chứng minh an toàn hơn. Cả tám câu cố định còn có token `tool_call` lộ ra đầu câu trả lời. Cần sửa mẫu chat hoặc bước giải mã và kiểm tra lại đầu ra trước khi dùng mô hình thực tế.

---

## 5. Đánh đổi theo β (bonus `make beta-sweep`)

Không chạy β-sweep, nên không có số liệu thực nghiệm cho β=0,05 và β=0,5. Giả thuyết của tôi là β=0,05 có thể cho policy đi xa reference hơn, tăng chênh lệch log-xác suất nhưng cũng tăng nguy cơ học theo nhãn nhiễu. Với β=0,5, mức phạt đi xa reference lớn hơn nên thay đổi hành vi có thể nhỏ hơn; margin định nghĩa là β nhân log-ratio nên không thể so các giá trị margin giữa các β nếu bỏ qua hệ số này. Cần huấn luyện lại cùng seed, dữ liệu, số bước và đánh giá held-out trước khi khẳng định giả thuyết.

---

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

Quyết định quan trọng nhất của tôi là chỉ dùng giám khảo vượt ngưỡng kiểm tra tiếng Việt 80% để báo win rate, thay vì giữ đủ hai reward model cho đẹp số liệu. Phương án thay thế là chấp nhận cả Qwen3 và Llama trong một hội đồng, hoặc gọi thêm một API judge độc lập. Cách đầu tiên tăng số giám khảo nhưng không khắc phục việc Qwen3 xếp đúng chỉ 8/12 cặp hiển nhiên; cách thứ hai cần thêm chi phí và một quy trình kiểm tra riêng. Tôi chọn ngưỡng đã ghi trong notebook trước khi thấy kết quả, dùng Llama vì nó đạt 12/12. Với 50 câu held-out, Llama chấm DPO thắng 12, SFT thắng 9, hoà 29. Win rate 0,53 có khoảng tin cậy 95% từ 0,44 đến 0,62, nên quyết định này không tạo ra bằng chứng cải thiện rõ ràng. Điểm bất ngờ là hai giám khảo đồng ý tới 87,9% trên 58 câu, nhưng Qwen3 vẫn trượt bài sanity; mức đồng thuận cao không thay thế được kiểm tra chất lượng giám khảo. Nếu làm lại, tôi sẽ sửa token `tool_call` lộ trong đầu ra, thêm một giám khảo khác nhà phát triển và mở rộng bộ sanity. Tôi cũng sẽ chấm tay vài câu có ràng buộc chặt, như chỉ dùng gạo và trứng, vì reward model có thể thích một câu trôi chảy nhưng sai điều kiện đầu vào.

---

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

Không chạy NB6. Không có số liệu IFEval, GSM8K hoặc Global-MMLU-vi để kết luận về khả năng làm theo chỉ dẫn, suy luận toán hay thuế căn chỉnh.

---

## 8. Biến thể loss (bonus NB3b)

Không chạy NB3b. Bài này chỉ có số liệu cho DPO cơ sở, chưa đủ để xếp hạng RPO, DPO-norm, LD-DPO và ORPO theo độ dài câu trả lời.

---

## 9. GRPO (bonus NB7)

Không chạy NB7. Không có đường reward hoặc độ chính xác trước/sau để kết luận về GRPO.

---

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] `BONUS-CHALLENGE.md` (không chấm điểm)

---

## Điều bất ngờ nhất

Điều bất ngờ nhất là DPO tạo margin dương trên held-out nhưng giám khảo vẫn cho 29/50 câu hoà và khoảng tin cậy chứa 0,5. Biểu đồ học được sở thích tương đối không thay thế cho kiểm tra chất lượng câu trả lời.
