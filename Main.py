#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/12/10 13:52        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：程序界面运行主入口 √ ━━━━━☆*°☆*°
"""
from Tkinter import *

import os
import tkFileDialog
from ttk import *
import Train_process
import thread
from datetime import datetime
from tkMessageBox import *

# 是否已经开始训练
train_flag = False
# 决策树深度默认为3
depth_default = 3
# 弱分类器数量默认为5
num_default = 6
# 训练样本正例文件夹
folder_train = ''
# 训练样本负例文件夹
folder_train_neg = ''
# 训练模型路径
model_dir = ''
# 训练信息输出迭代器
global train_generator

# 往表格里插入信息
def insert_info(texts):
	l.insert('',0, values=texts)


# 递归视图更新，每隔20ms进行一次(以守护进程的形式)
def update_info():
	global train_generator, train_flag
	try:
		info = train_generator.next()
		if info == 'ok':
			pos = 0
			neg = 0
			pre_result = train_generator.next()
			for root, dirs, files in os.walk(folder_train + '/', topdown=False):
				for i in range(len(files)):
					if pre_result[i] == 1:
						pos += 1
						insert_info([str(i) + '-' + files[i] + '-' + 'Positive', datetime.now()])
					else:
						neg += 1
						insert_info([str(i) + '-' + files[i] + '-' + 'Negative', datetime.now()])
			sums = pos + neg
			str_ = ("对测试集的%d个样本进行预测，统计结果为:" % sums) + "\n预测负例有" + str(neg) + '个, \n预测正例有' + str(pos) + '个\n具体对应请看日志输出'
			showinfo("预测结果", str_)
		elif info == 'Finish':
			train_flag = False
			pre_accuracy = train_generator.next()
			str_ = str(pre_accuracy)
			showinfo("模型在1/5验证集上的效果评估：", str_)
			pre_accuracy = train_generator.next()
			str_ = str(pre_accuracy)
			showinfo("模型在全局上的效果评估：", str_)
		else:
			insert_info([info, datetime.now()])
	except:
		pass
	t.after(20, update_info)

# 得到训练的过程信息迭代器
def get_genarator(depth, num, pos_dir, neg_dir):
	global train_generator
	train_generator = Train_process.train_main(depth, num, pos_dir, neg_dir)

# 进程开启函数
def thread_function(option):
	if option == 1:
		t.after(20, update_info)
	# 这个用于调试而已，实际上训练过程在主进程中进行
	elif option == 2:
		t.after(20, get_genarator(depth_default, num_default, folder_train))
	elif option == 3:
		pop_up_box()

# 弹出选择模型参数的窗口 + 训练开启
def pop_up_box():
	def inputint():
		global train_flag
		if 0 < int(var_num.get()) <= 10 and 0 < int(var_depth.get()) <= 10:
			if askyesno('确认提示', '确定开始进行模型训练？') and not train_flag and not folder_train == '' and not folder_train_neg == '':
				c_root.destroy()
				train_flag = True
				# 主进程进行训练
				get_genarator(var_depth.get(), var_num.get(), folder_train + '/', folder_train_neg + '/')
			elif train_flag:
				showerror("错误提示", "已经开启训练，请训练结束后再次开启")
				c_root.destroy()
			elif folder_train == '' or folder_train_neg == '':
				showerror("错误提示", "请先选择训练目录(需要保证全部是图片)")
				c_root.destroy()
		else:
			showerror("错误提示", "请保证两个参数都满足0<x<=10")

	def inputclear():
		var_depth.set('')

	c_root = Tk(className='输入模型参数')
	c_root.geometry('220x60')
	f = Frame(c_root)
	f.pack(side=TOP, expand=YES, fill=BOTH, anchor=NW)
	label1 = Label(f, text='决策树深度：')
	label1.pack(side=LEFT)
	# 记住，var变量一定要跟所在根绑定！bug修复*2
	var_depth = StringVar(f)
	entry1 = Entry(f, textvariable=var_depth, width=5)
	var_depth.set('3')
	entry1.pack(side=LEFT)
	btn1 = Button(f, text='Summit', command=inputint)
	btn1.pack(side='right')
	ff = Frame(c_root)
	ff.pack(side=BOTTOM, expand=YES, fill=BOTH, anchor=NW)
	label2 = Label(ff, text='弱分类器数量：')
	label2.pack(side=LEFT)
	var_num = StringVar(ff)
	entry2 = Entry(ff, textvariable=var_num, width=5)
	var_num.set('5')
	entry2.pack(side=LEFT)
	entry2['state'] = 'readonly'
	btn2 = Button(ff, text='Reset', command=inputclear)
	btn2.pack(side='right')
	c_root.mainloop()

# 通过选择的模型对测试集进行测试
def predict():
	global train_generator
	if askyesno('确认提示', '确定使用这个模型进行图片预测？') and not train_flag and not folder_train == '' and not model_dir == '':
		train_generator = Train_process.test_predict(model_dir, folder_train + '/')
	elif train_flag:
		showerror("错误提示", "已经开启训练，请训练结束后再进行预测")
	elif folder_train == '':
		showerror("错误提示", "测试集目录不能为空")
	elif model_dir == '':
		showerror("错误提示", "训练模型路径不能为空")

if __name__ == '__main__':
	t = Tk()
	t.title('Adaboost_V v1.1')
	t.geometry('580x400')
	t.iconbitmap(r'me.ico')  # 设置图标
	f1 = Frame(t)
	f1.pack(side=TOP, fill=X, anchor=N)
	var = StringVar()
	var.set('请选择图片训练集(正负两个目录)/测试集目录')
	e2 = Entry(f1, textvariable=var)
	e2.pack(side=LEFT, expand=YES, fill=X)
	e2['state'] = 'readonly'
	b = Button(f1, text='开始训练', command=lambda: thread_function(3))
	b.pack(side=RIGHT, anchor=W)
	def set_directiory(var):
		global folder_train, folder_train_neg
		if askyesno('选择提示', '选择"是"则开始选取模型的训练正负样本目录(先后2个)\n选择“否”则选择测试文件夹目录)'):
			folder_train = ''
			folder_train_neg = ''
			folder_train = tkFileDialog.askdirectory(title=u"选择存放训练正例图片的目录")
			folder_train_neg = tkFileDialog.askdirectory(title=u"选择存放训练负例图片的目录")
			var.set('Train: '+folder_train + ' + ' + folder_train_neg)
		else:
			folder_train = ''
			folder_train_neg = ''
			folder_train = tkFileDialog.askdirectory(title=u"选择存放测试图片的目录")
			var.set('Test: ' + folder_train)
	b2 = Button(f1, text='选择文件夹', command=lambda: set_directiory(var))
	b2.pack(side=TOP)
	f3 = Frame(t)
	f3.pack(side=TOP, fill=X, anchor=N)
	var2 = StringVar()
	var2.set('请选择训练模型准确路径')
	e2 = Entry(f3, textvariable=var2)
	e2['state'] = 'readonly'
	e2.pack(side=LEFT, expand=YES, fill=X)
	b3 = Button(f3, text='开始预测', command=predict)
	def set_model(var2):
		global model_dir
		model = tkFileDialog.askopenfilename(title=u"选择已训练好的模型")
		var2.set(model)
		model_dir = model
	b3.pack(side=RIGHT, anchor=W)
	b4 = Button(f3, text='选择模型', command=lambda: set_model(var2))
	b4.pack(side=TOP)
	f2 = Frame(t)
	f2.pack(side=TOP, expand=YES, fill=BOTH, anchor=NW)
	l = Treeview(f2, columns=('Info', 'Time'), show='headings')
	l.column('Info', width=330, anchor='w')
	l.column('Time', width=30, anchor='e')
	l.heading('Info', text='Info')  # 对列名命名
	l.heading('Time', text='Time')
	l.pack(side=LEFT, fill=BOTH, expand=YES, anchor=NW)
	scb = Scrollbar(l)
	scb.config(command=l.yview)
	l.config(yscrollcommand=scb.set)
	scb.pack(side=RIGHT, expand=YES, fill=Y, anchor=E)
	# 开启更新视图线程(守护进程)
	thread.start_new_thread(thread_function, (1,))
	# 训练进程在pop_up_box()里面开启
	t.mainloop()