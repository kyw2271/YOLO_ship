from tkinter import *
import tkinter.ttk as ttk

#파일을 가져오기 위한 모듈
from tkinter import filedialog
from PIL import ImageTk,Image



root = Tk() 
#gui이름
root.title("gui test project")

#파일 오픈 , 종료  버튼 만들기
file_button_frame = Frame(root)
file_button_frame.pack(fill="x", padx=5, pady=5)

def file_open_fun():
  files = filedialog.askopenfilenames(title= "이미지 파일을 선택하세요",\
    filetypes= (("PNG 파일", "*.png"), ("모든 파일", "*.*")),
    #최초에 보여줄 dir를 명시
    #r을 쓰면 넣어준 문자그대로를 경로로 사용
    initialdir=r"D:\파이선GUI공부\guiproject\예시사진"
    )
    #사용자가 선택한 파일목록 출력
  for file in files:
    #이제 리스트박스에 출력을 해주면됨 
    list_file.insert(END,file)





file_open_button = Button(file_button_frame,text="파일열기", padx=5, pady=5, width=10, command=file_open_fun)
file_open_button.pack(side="left")

file_exit_button = Button(file_button_frame,text="종료", padx=5, pady=5, width=10, command=root.quit)
file_exit_button.pack(side="right")




#------------------------------------------------------------------------------------------------------------------

#2-1
#오픈된 파일 경로 표시 

#list프레임 그리고 스크롤바
list_frame = Frame(root)
#화면 전체에 펴지도록 프레임을 both로 채움
list_frame.pack(fill="both", padx=5, pady=5)

#스크롤바 구현
scrollbar= Scrollbar(list_frame)
#스크롤바는 리스트 프레임 오른쪽에 그리고 y축으로 쭉 늘리기
scrollbar.pack(side="right", fill="y")

#리스트박스를 실제 구현
#리스트 프레임 넣고, 높이는 15면 한번에 15개의 파일을 보고, 스크롤바와 연동하기 위해 yscrollcommand=scrollbar.set으로 스크롤바와 매핑-1
list_file = Listbox(list_frame, selectmode="extended", height =15, yscrollcommand=scrollbar.set)
list_file.pack(side="left",fill ="both", expand=True)

#스크롤바 -> 리스트파일과 mapping-2
scrollbar.config(command=list_file.yview)


#------------------------------------------------------------------------------------------------------------------

#파일 경로 저장 배열

file_path_arr=[]


#3-1선택한 파일 경로 넘기기 버튼 

#3-2선택된 파일 삭제 버튼


select_file_button_frame = Frame(root)
select_file_button_frame.pack(fill="both", padx=5, pady=5)


#3-1 선택한 파일 경로 넘기기 버튼 함수

def file_send_func():
  for index in list_file.curselection():
    #print(list_file.get(index))
    file_path_arr.append(list_file.get(index))



    

#3-2 선택된 파일 삭제 버튼 함수
def file_del_func():

  #print(list_file.curselection())
  #보통 삭제시 앞에서부터 지우게되면 index가 하나씩 앞으로 당겨지므로 뒤에서부터 지움
  #reverse()는 리스트를 아예바꿈
  #revesed()는 바뀐 리스트를 반환
  for index in reversed(list_file.curselection()):
    list_file.delete(index)


#3-3 확인용

def file_path():
  for elem in file_path_arr:
    print(elem)

    

#3-1선택한 파일 경로 넘기기 버튼 

#send button
file_send_button = Button(select_file_button_frame,text="파일 경로 보내기", padx=5, pady=5, width=15, command=file_send_func)
file_send_button.pack(side="left")


#3-2선택된 파일 삭제 버튼
file_del_button = Button(select_file_button_frame,text="선택 삭제", padx=5, pady=5, width=15, command=file_del_func)
file_del_button.pack(side="right")



#3-3 경로저장 확인 
file_path_button = Button(select_file_button_frame,text="경로 확인", padx=5, pady=5, width=15, command=file_path)
file_path_button.pack(side="right")



#------------------------------------------------------------------------------------------------------------------

#4-1 선택된 사진 보기 
photo_frame = Frame(root)
photo_frame.pack(fill="x", padx=5, pady=5)

def photo_open():
  
  my_img = PhotoImage(file = "D:\파이선GUI공부/check.png",width=10,height=10)
  my_label = Label(root, image=my_img)
  my_label.pack(expand=1, anchor=CENTER)

  print(my_img)





photo_open_button = Button(photo_frame,text="사진 보기", padx=5, pady=5, width=10, command=photo_open)
photo_open_button.pack(side="left")


# photo_label_frame= LabelFrame(root, text="사진")
# photo_label_frame.pack(fill="x", padx=100, pady=100,ipady=200)




root.geometry("1600x1024")
root.mainloop()