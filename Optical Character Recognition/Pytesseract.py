from PIL import Image
import pytesseract


def ocr(filename):
    text = pytesseract.image_to_string(Image.open(filename))
    return text


def main():
    filename = 'test.png'
    print(ocr(filename))


if __name__ == '__main__':
    main()


print(pytesseract)
