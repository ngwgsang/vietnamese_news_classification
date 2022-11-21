import pickle
from sklearn.svm import LinearSVC, SVC
import sys

# Loại bỏ warning
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")



with open('svm_classifier_2.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    cv = pickle.load(f)


def predict_label_from_news(news):
    # Hàm này sẽ trả về nhãn của 1 bài viết
    id2label = {
        0: 'thời sự',
        1: 'thế giới',
        2: 'kinh doanh',
        3: 'thể thao',
        4: 'giải trí',
        5: 'pháp luật',
        6: 'sức khỏe',
        7: 'giáo dục',
        8: 'khoa học - công nghệ',
        9: 'du lịch - ẩm thực',
        10: 'oto xe máy'
    }
    return id2label[model.predict(cv.transform([news]).toarray())[0]]


# news = 'Loại thức uống quen thuộc này chính là nguyên nhân gây ra cái chết của vị giám đốc trẻ. Gan là một cơ quan quan trọng của cơ thể con người, chịu trách nhiệm chính cho các chức năng chuyển hóa và giải độc của cơ thể. Nếu khỏe mạnh, thì con người sẽ cảm thấy tràn đầy năng lượng, ngược lại sẽ cảm thấy mệt mỏi và suy nhược. Đặc điểm chính của bệnh gan là mệt mỏi, ban đầu nhiều người thường không để ý đến dấu hiệu này và bỏ qua vì nghĩ rằng do cơ thể không được nghỉ ngơi. Thế nhưng, chính sự chủ quan này đã khiến người bệnh bỏ lỡ thời điểm điều trị tốt nhất, khiến bệnh cứ thế phát triển và gây hại cho sức khỏe.Trên thực tế, sức khỏe của lá gan cần được quan tâm chăm sóc về mọi mặt thì mới có thể đẩy lùi được các loại bệnh tật. Tuy nhiên, nhiều người vì nhiều lý do mà thường xuyên phạm phải những thói quen độc hại ảnh hưởng xấu đến sức khỏe. Việc duy trì lối sống không lành mạnh này không sớm thì muộn sẽ là "sát thủ thầm lặng" cướp đi tính mạng bạn. Trường hợp dưới đây là một ví dụ.'
# label = predict_label_from_news(news)
# print(label)
# >>> sức khỏe