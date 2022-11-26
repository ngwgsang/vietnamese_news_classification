from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory='templates')


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



@app.post('/predict')
def disable_cat(request: Request, news: str = Form(...)):
    label = predict_label_from_news(news)
    return templates.TemplateResponse("index.html", {"request": request, "label": label})




@app.get('/', response_class=HTMLResponse)
def main(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})



    