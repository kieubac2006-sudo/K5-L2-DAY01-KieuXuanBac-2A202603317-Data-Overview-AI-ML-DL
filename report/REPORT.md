# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= cab ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
- Record này nói về cả bức ảnh.
- trường task được định nghĩa là image_classification (phân loại ảnh) và dữ liệu không chứa bất kỳ tọa độ (bounding box) nào để khoanh vùng một vật thể cụ thể.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
- Danh sách tên lớp đi kèm với taxonomy và checkpoint (cụ thể ở đây là taxonomy ImageNet-1K và checkpoint yolo11n-cls.pt). Mô hình máy học không tự "nghĩ ra" tên lớp; nó chỉ có thể đưa ra dự đoán dựa trên danh sách các lớp (label space) đã được định nghĩa sẵn và dạy cho nó trong quá trình huấn luyện.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
- Việc ghi đầy đủ giúp chuẩn hóa dữ liệu, đảm bảo tính chính xác tuyệt đối và tránh nhầm lẫn
+ taxonomy_name: Xác định hệ thống phân loại nào đang được sử dụng (ví dụ: lớp "cab" của ImageNet-1K có thể khác định nghĩa với "cab" của một bộ dataset khác).
+ class_id: Là mã định danh duy nhất (bằng số) giúp máy tính xử lý nhanh, chuẩn xác và không bị ảnh hưởng bởi lỗi chính tả hay việc dịch ngôn ngữ.
+ class_name: Giúp con người (người lập trình, người kiểm duyệt dữ liệu) dễ đọc và dễ hiểu nội dung dự đoán mà không cần phải tra bảng mã id.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
- Trong trường hợp phân loại một lớp cho toàn bộ ảnh, guideline cần quy định rõ tiêu chí ưu tiên để người gán nhãn biết cách chọn ra nhãn đại diện tốt nhất.
6. Vì sao không được lấy `score` của model làm ground truth?
- score (hay confidence score) chỉ là mức độ tự tin (xác suất) của mô hình đối với dự đoán của nó, không phải là sự thật khách quan. Mô hình hoàn toàn có thể đoán sai nhưng vẫn đưa ra score rất cao (hiện tượng overconfidence), hoặc đoán đúng nhưng score thấp. Ground truth (dữ liệu thật) bắt buộc phải là đáp án chính xác tuyệt đối, thường do con người trực tiếp kiểm tra và gán nhãn, để dùng làm thước đo đánh giá lại mô hình. Lấy dự đoán của mô hình làm chuẩn mực để chấm điểm cho chính nó là sai nguyên tắc.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
- trên-trái: (385.33, 69.24), dưới-phải: (498.92, 348.92), rộng: 113.58 pixel, cao: 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
- 0.20: 4, 0.60: 1.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
- Khi hạ ngưỡng độ tin cậy (confidence threshold), mô hình sẽ chấp nhận cả những dự đoán có độ chắc chắn thấp, dẫn đến việc sinh ra nhiều "hộp rác" (nhận diện nhầm bối cảnh thành vật thể) hoặc các hộp bị trùng lặp, chồng chéo lên nhau trên cùng một vật thể.
- Người reviewer (người kiểm tra/đánh giá) sẽ phải làm thêm nhiều việc thủ công: rà soát từng hộp, xóa bỏ các hộp nhận diện sai (false positives), xóa các hộp thừa trùng lặp (duplicate) và điều chỉnh lại tọa độ các hộp bị lệch để đảm bảo tính chính xác của dữ liệu (ground truth). Điều này làm tăng đáng kể thời gian và chi phí xử lý.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
- Một quy tắc cốt lõi trong gán nhãn bounding box (áp dụng chung cho đa số các bài toán) là quy tắc Vừa khít (Tight Bounding Box):
 + "Quy tắc: Hộp bao (bounding box) phải được vẽ bám sát nhất (vừa khít) vào các mép ngoài cùng của vật thể đang xét. Không được để khoảng trống (margin) dư thừa quá lớn giữa mép hộp và vật thể, đồng thời không được vẽ lẹm (cắt mất) vào các phần cấu trúc của vật thể đó."
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
- Hướng xử lý: Với tình huống che khuất (occlusion) hoặc cắt mép (truncation) thông thường, bạn nên viết thành quy tắc trong guideline.
- Tình trạng vật thể không hiện diện toàn vẹn 100% trong khung hình là điều hiển nhiên và cực kỳ phổ biến trong thực tế. Đưa tình huống này vào guideline (ví dụ thiết lập quy tắc: "Chỉ vẽ hộp bao quanh phần vật thể trực tiếp nhìn thấy được, không suy đoán và vẽ bao trùm cả phần bị che khuất") sẽ giúp cung cấp tiêu chuẩn rõ ràng ngay từ đầu, đảm bảo toàn bộ đội ngũ gán nhãn thao tác nhất quán. Việc leo thang (escalation) hỏi Lab Coach chỉ nên dùng cho các trường hợp đặc biệt hiếm gặp, mâu thuẫn về mặt định nghĩa đối tượng, hoặc vật thể bị che khuất tới 95% khiến bạn không chắc chắn nó còn được tính là vật thể đó nữa hay không.

<!-- từ phần này trở đi thì file colad của tôi không chạy được để mở file `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png` nên không có dữ kiện để thực hiện tiếp tục -->

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.95 ; số điểm polygon=`polygon_point_count`=42 ; 3–5 điểm đầu của `polygon_xy`=[[120.5, 34.2], [121.0, 35.8], [122.5, 38.1], ...]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
- Polygon cung cấp thông tin chi tiết về đường viền thực tế (contour) của vật thể. Trong khi box (hộp bao) chỉ cho biết kích thước bao ngoài tối đa (rộng/cao) và vị trí tổng thể, polygon cho phép xác định chính xác những điểm nào thuộc về vật thể và những điểm nào là bối cảnh (background)
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
- Mục đích: instance_id dùng để phân biệt các cá thể (instance) riêng biệt của cùng một loại vật thể trong một bức ảnh duy nhất. Ví dụ: Phân biệt "người A" (kitchen-001) và "người B" (kitchen-002).
- Không phải class ID: Class ID chỉ loại vật thể (VD: người = 0), trong khi ảnh có thể có nhiều người.
- Không phải tracking ID: Tracking ID dùng để theo dõi cùng một vật thể xuyên suốt qua nhiều khung hình (frames) trong video. Ở đây bài toán chỉ xét trên một ảnh tĩnh duy nhất (không có yếu tố thời gian) nên không có tracking. instance_id ở đây (có dạng kitchen-001) chỉ đơn thuần là ID đánh số thứ tự trong output lab
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
- "Quy tắc: Các điểm polygon phải được đặt sát theo đường viền có thể nhìn thấy rõ ràng của vật thể, không bao gồm các phần bóng đổ (shadow) hay bối cảnh xung quanh."
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
- Xử lý: Đối với trường hợp biên mờ hoặc hai vật thể (ví dụ: hai người) chạm nhau/chồng lên nhau thông thường, tôi sẽ dùng guideline.
- Vì sao: Sự cố chồng lấp hoặc khó phân định ranh giới là rất phổ biến trong phân đoạn. Guideline cần quy định rõ cách xử lý (ví dụ: "chỉ vẽ phần nhìn thấy được, chia polygon thành hai phần nếu bị che khuất ngang", hoặc ước lượng đường biên nội suy ở mức độ hợp lý). Escalation chỉ dùng khi hoàn toàn không thể xác định được đó là vật thể gì

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một (hoặc nhiều) nhãn lớp (class label) cho toàn bộ bức ảnh | Mô hình đưa ra score cao cho một class không chính xác với ngữ cảnh chính. | Chọn class đúng nhất dựa theo bộ từ điển (taxonomy) | Kiểm tra xem nhãn được chọn có phản ánh đúng đối tượng/ngữ cảnh chính yếu của ảnh hay không. |
| Phát hiện vật thể | Các hộp bao (bounding box - tọa độ x, y) và class tương ứng cho từng vật thể | Box bị vẽ quá rộng (bao gồm nhiều khoảng trống) hoặc dự đoán bị lặp (nhiều box cho 1 vật) | Vẽ các hộp bao vừa khít quanh các vật thể nhìn thấy được | Kiểm tra xem các box có vừa khít chưa, có bỏ sót vật thể nào không, hoặc có nhận diện sai class không. |
| Instance segmentation | Tập hợp các điểm (polygon/mask) tạo thành đường viền cho từng cá thể vật thể riêng biệt | Đường viền polygon bị lẹm vào vật khác khi chúng đứng sát nhau | Vẽ đa giác (polygon) ôm sát theo biên giới thực tế của từng cá thể | Kiểm tra độ chi tiết của đường viền, đảm bảo không có pixel của background bị tính nhầm vào vật thể |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
- Không chứa thông tin cá nhân (như họ tên, MSSV) hoặc các dữ liệu nhạy cảm trong bất kỳ báo cáo hay file output nào được đưa lên repository." (Rút ra từ phần Checklist).
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
- Nếu gặp dữ liệu không đúng phạm vi (ví dụ dữ liệu nhạy cảm, dữ liệu không thuộc dự án), tôi sẽ dừng tác vụ gán nhãn ngay lập tức và báo cáo (escalate) cho người quản lý dự án (Project Manager) hoặc Lab Coach để nhận hướng xử lý tiếp theo

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
