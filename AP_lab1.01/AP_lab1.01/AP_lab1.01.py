import json
import sys
import xml.etree.ElementTree as ET
import xml

class Book():
    def __init__(self, title, author, price):
        self._title = title
        self._author = author
        self._price = price

    def dataParser(self, data):
        try:
            self._title = data["Book"]["_title"]
            self._author = data["Book"]["_author"]
            self._price = data["Book"]["_price"]
        except KeyError:
            print("Wrong data")
            sys.exit()

    def getTitle(self):
        return self._title

    def setTitle(self, title):
        self._title = title

    def getAuthor(self):
        return self._author

    def setAuthor(self, author):
        self._author = author

    def getPrice(self):
        return self._price

    def setPrice(self, price : int):
        self._price = price


class eBook(Book):

    def __init__(self, title, author, price, size):
        super().__init__(title, author, price)
        self._size = size

    def dataParser(self, data):
        try:
            self._title = data["eBook"]["_title"]
            self._author = data["eBook"]["_author"]
            self._price = data["eBook"]["_price"]
            self._size = data["eBook"]["_size"]
        except KeyError:
            print("Wrong data")
            sys.exit()

    def getsize(self):
        return self._size

    def setsize(self, size : int):
        self._size = size

class AudioBook(Book):
    def __init__(self, title, author, price, duration):
        super().__init__(title, author, price)
        self._duration = duration

    def dataParser(self, data):
        try:
            self._title = data["AudioBook"]["_title"]
            self._author = data["AudioBook"]["_author"]
            self._price = data["AudioBook"]["_price"]
            self._duration = data["AudioBook"]["_duration"]
        except KeyError:
            print("Wrong data")
            sys.exit()

    def getDuration(self):
        return self._duration

    def setDuration(self, duration : int):
        self._duration = duration

def JSONserializ(object, fileName):
    classObj = str(object.__class__).replace("'", ".").split(".")[2]
    with open(fileName, "w") as f:
        json.dump({classObj : object.__dict__}, f)


def JSONdeserializ(fileName):
    try:
        with open(fileName) as f:
            dataList = json.loads(f.read())
        classObj = str(dataList.keys()).split("'")[1]

        if classObj == "Book":
            tBook.dataParser(dataList)
        elif classObj == "eBook":
            teBook.dataParser(dataList)
        elif classObj == "AudioBook":
            tAudioBook.dataParser(dataList)
        else:
            return "Wrong data"

    except FileNotFoundError:
        print("File doesn't exist")
        sys.exit()
    return dataList

def XMLserializ(object, fileName):
    classObj = str(object.__class__).replace("'", ".").split(".")[2]
    tempPat = ET.Element(classObj)
    dictObj = object.__dict__
    for elem in dictObj:
        if elem == '_id':
            tempPat.set("id", str(object.getId()))
        v = ET.SubElement(tempPat, elem)
        v.text = str(dictObj.get(elem))

    s = ET.tostring(tempPat, encoding="utf-8", method="xml")
    s = s.decode("UTF-8")
    with open(fileName, "w") as f:
        f.write(s)
        f.close()

def XMLdeserializ(fileName):
    try:

        li = ET.parse(fileName).getroot()
        classObj = str(li).split("'")[1]
        dictObj = {}
        for el in li:
            dictObj[el.tag] = el.text

        if classObj == "Book":
            tBook.dataParser({classObj : dictObj})
        elif classObj == "eBook":
            teBook.dataParser({classObj : dictObj})
        elif classObj == "AudioBook":
            tAudioBook.dataParser({classObj : dictObj})
        else:
            return "Wrong data"

        return {classObj : dictObj}
    except FileNotFoundError:
        print("File not found")
    except xml.etree.ElementTree.ParseError: 
        print("File incorrected")
    except Exception:
        print("Error")

def printMenu():
    print("\n 1 - fill object")
    print(" 2 - output object")
    print(" 3 -   serialization XML")
    print(" 4 - deserialization XML")
    print(" 5 -   serialization JSON")
    print(" 6 - deserialization JSON")
    print(" 0 - complete the program\n")

tempTitle = ""
tempAuthor = ""
tempPrice = ""
tempSize = ""
tempDuration = ""

tBook = Book("None", "None", "None")
teBook = eBook("None", "None", "None", "None")
tAudioBook = AudioBook("None", "None", "None", "None")
temp = Book("None", "None", "None")

n = ""
while n != "0":
    printMenu()
    n = input("Choice: ")
    if n == "1":
        print("Select object class:\n1 - Book\n2 - eBook\n3 - AudioBook\n")
        x = ''
        while True:
            x = input("Class = ")
            if (x == "1") or (x == "2") or (x == "3"):
                break
            else: 
                print('Input Error.')
        tempTitle = input("Title = ")
        tempAuthor = input("Author = ")
        tempPrice = input("Price = ")
        if x == "1":
            temp = Book(tempTitle, tempAuthor, tempPrice)
        elif x == "2":
            tempSize = input("Size = ")
            temp = eBook(tempTitle, tempAuthor, tempPrice, tempSize)
        elif x == "3": 
            tempDuration = input("Duration = ")
            temp = AudioBook(tempTitle, tempAuthor, tempPrice, tempDuration)
    elif n == "2":
        print({str(temp.__class__).replace("'", ".").split(".")[2] : temp.__dict__})
    elif n == "3":
        f = input("Enter file name: ")
        XMLserializ(temp, f+".xml")
    elif n == "4":
        f = input("Enter file name: ")
        print(XMLdeserializ(f+".xml"))
    elif n == "5":
        f = input("Enter file name: ")
        JSONserializ(temp, f+".json")
    elif n == "6":
        f = input("Enter file name: ")
        print(JSONdeserializ(f+".json"))

