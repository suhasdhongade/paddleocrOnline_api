## PaddleOCR FastAPI For RPA Projects

This project is a **FastAPI-based OCR (Optical Character Recognition) API** using **PaddleOCR**. It allows you to extract text from images and returns the detected text along with bounding boxes.

---

### **🚀 Features**

- Detect and extract text from images
- Return text along with bounding box coordinates
- Save images with detected text overlays
- FastAPI-based RESTful API

---

## **🔧 Installation**

### **Step 1: Clone the Repository**

```sh
git clone https://github.com/your-username/paddleocr_api.git
cd paddleocr_api
```

### **Step 2: Create a Virtual Environment**

```sh
python -m venv venv  # For Windows
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate  # For Windows (Command Prompt)
```

### **Step 3: Install Dependencies**

```sh
pip install -r requirements.txt
```

---

## **🚀 Running the API**

Start the FastAPI server with:

```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://127.0.0.1:8000/docs** to access the interactive Swagger UI.

---

## **🖼️ Usage**

### **Upload an Image for OCR**

**Endpoint:** `POST /upload`

**Example (Using `curl`)**:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_image.jpg'
```

**Response:**

```json
{
  "text": "Detected text here",
  "boxes": [[x1, y1, x2, y2], [x3, y3, x4, y4]]
}
```

---

## **📂 Project Structure**

```

├── ocr_api.py           # FastAPI application
├── requirements.txt  # Required dependencies
├── README.md         # Project documentation
```

---

## **📜 License**

This project is licensed under the MIT License.

---

## **🤝 Contributing**

Feel free to submit a pull request or open an issue if you have suggestions or improvements!

---

## **👨‍💻 Author**

**Suhas Dhongade**
