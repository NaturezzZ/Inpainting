﻿#pragma warning (disable:4996)
#include <easyx.h>
#include <fstream>
#include <conio.h>
#include <queue>
#include <vector>
#include <graphics.h>
#include <iostream>
#include <stdlib.h>
#include <Windows.h>
#include <algorithm>
#include "math.h"
#include <string>
#define pi 3.1415926535
using namespace std;
TCHAR str1[] = _T("Image Inpainting Project");
TCHAR str2[] = _T("在此插入图片");
TCHAR str3[] = _T("图片补全");
TCHAR str4[] = _T("涂");
TCHAR str5[] = _T("鸦");
TCHAR str6[] = _T("笔");
TCHAR str7[] = _T("by 郑乃千，杨君维，李典泽");
TCHAR str8[] = _T("图库中无该图片，请重新选择！");
TCHAR str9[] = _T("Waiting");
TCHAR str10[] = _T("涂改后");
TCHAR str11[] = _T("原图");
TCHAR str12[] = _T("修复结果");
TCHAR str13[] = _T("另存为");
TCHAR str14[] = _T("返回");
TCHAR str15[] = _T("https://github.com/NaturezzZ/Inpainting");
TCHAR str16[] = _T("退出");
IMAGE background, picture, finalpicture;
struct ccircle {
	int x;
	int y;
	int radius;
	ccircle(int _x, int _y, int _r) :x(_x), y(_y), radius(_r) {}
};
vector<ccircle> mask;
vector<IMAGE> imask;
int mmask[256][256];
void iloadimage() {
	loadimage(&background, _T("setbkinit.jpg"));
}
void setbkinit() {
	putimage(0, 0, &background);
}
void setpictureinit() {
	putimage(369, 100, &picture);
}
void setpicinit() {
	setlinestyle(PS_DASH | PS_ENDCAP_FLAT, 3);
	setlinecolor(LIGHTCYAN);
	rectangle(369, 100, 626, 356);
	setlinestyle(PS_SOLID | PS_ENDCAP_ROUND, 3);
	line(485, 200, 510, 200);
	line(497, 188, 497, 212);
	setlinestyle(PS_SOLID | PS_ENDCAP_FLAT);
	setlinecolor(BLUE);
	roundrect(370, 400, 625, 510, 20, 20);
	setfillcolor(BLUE);
	floodfill(498, 455, BLUE);
	setfillcolor(MAGENTA);
	setlinecolor(MAGENTA);
	fillrectangle(635, 100, 665, 300);
	setfillcolor(WHITE);
	setlinecolor(WHITE);
	fillcircle(650, 225, 15);
	fillcircle(650, 265, 10);
	fillcircle(650, 295, 5);
}
void settextinit() {
	settextstyle(150, 20, _T("Zapfino"), 0, 0, FW_EXTRABOLD, true, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	settextcolor(LIGHTBLUE);
	setbkmode(TRANSPARENT);
	outtextxy(250, 0, str1);
	settextstyle(30, 0, _T("锐字云字库隶书体1.0"), 0, 0, FW_EXTRALIGHT, false, false, false);
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	settextcolor(LIGHTCYAN);
	outtextxy(420, 230, str2);
	settextcolor(YELLOW);
	gettextstyle(&f);
	f.lfHeight = 70;
	settextstyle(&f);
	outtextxy(375, 430, str3);
	gettextstyle(&f);
	f.lfHeight = 35;
	settextstyle(&f);
	outtextxy(635, 100, str4);
	outtextxy(635, 125, str5);
	outtextxy(635, 150, str6);
}
void finaltext() {
	settextstyle(40, 0, _T("宋体"), 0, 0, FW_BOLD, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	settextcolor(BLACK);
	setbkmode(TRANSPARENT);
	outtextxy(175, 370, str10);
	outtextxy(457, 370, str11);
	outtextxy(680, 370, str12);
}
void setfinalinit() {
	setbkinit();
	putimage(107, 100, &finalpicture);
	settextstyle(150, 20, _T("Zapfino"), 0, 0, FW_EXTRABOLD, true, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	settextcolor(LIGHTBLUE);
	setbkmode(TRANSPARENT);
	outtextxy(250, 0, str1);
	finaltext();
	settextstyle(30, 0, _T("宋体"), 0, 0, FW_BOLD, true, false, false);
	settextcolor(BLACK);
	outtextxy(600, 670, str7);
	settextstyle(40, 0, _T("宋体"), 0, 0, FW_BOLD, false, false, false);
	settextcolor(WHITE);
	setbkmode(OPAQUE);
	setbkcolor(CYAN);
	outtextxy(310, 500, str13);
	outtextxy(570, 500, str14);
}
void ifsave() {
	settextstyle(40, 0, _T("宋体"), 0, 0, FW_BOLD, false, false, false);
	settextcolor(RED);
	setbkmode(OPAQUE);
	setbkcolor(CYAN);
	outtextxy(310, 500, str13);
}
void ifback() {
	settextstyle(40, 0, _T("宋体"), 0, 0, FW_BOLD, false, false, false);
	settextcolor(RED);
	setbkmode(OPAQUE);
	setbkcolor(CYAN);
	outtextxy(570, 500, str14);
}
void iftransform() {
	settextstyle(30, 0, _T("锐字云字库隶书体1.0"), 0, 0, FW_EXTRALIGHT, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfHeight = 70;
	settextstyle(&f);
	settextcolor(RED);
	outtextxy(375, 430, str3);
}
void ifpicture() {
	settextstyle(30, 0, _T("锐字云字库隶书体1.0"), 0, 0, FW_EXTRALIGHT, false, false, false);
	LOGFONT f;
	setlinecolor(WHITE);
	setlinestyle(PS_DASH | PS_ENDCAP_FLAT, 3);
	rectangle(369, 100, 626, 356);
	setlinestyle(PS_SOLID | PS_ENDCAP_ROUND, 3);
	line(485, 200, 510, 200);
	line(497, 188, 497, 212);
	gettextstyle(&f);
	f.lfHeight = 30;
	settextstyle(&f);
	settextcolor(WHITE);
	outtextxy(420, 230, str2);
}
void clickcircle(int x, int y, int radius) {
	setlinecolor(BLUE);
	setfillcolor(BLUE);
	fillcircle(x, y, radius);
}
void followcircle(int x, int y, int radius) {
	setlinecolor(WHITE);
	setfillcolor(WHITE);
	fillcircle(x, y, radius);
}
bool incircle(int x1, int y1, int x2, int y2, int radius) {
	if (sqrt((x1 - x2)*(x1 - x2)*1.0 + (y1 - y2)*(y1 - y2)*1.0) < radius)
		return true;
	return false;
}
void tomask() {
	vector<ccircle>::iterator p;
	setfillcolor(WHITE);
	setlinecolor(WHITE);
	for (p = mask.begin(); p != mask.end(); p++)
		fillcircle(p->x, p->y, p->radius);
}
void topicture() {
	IMAGE opicture = imask.back();
	putimage(369, 100, &opicture);
}
void getmask(int x, int y, int radius) {
	for (int i = y - radius; i <= y + radius; i++)
		for (int j = x - radius; j <= x + radius; j++)
			if (incircle(i, j, y, x, radius) == true && i >= 0 && j >= 0 && i < 256 && j < 256)
				mmask[i][j] = 255;
}
void rechoose() {
	settextstyle(20, 0, _T("黑体"), 0, 0, FW_BLACK, false, false, false);
	settextcolor(RED);
	setbkcolor(YELLOW);
	setbkmode(OPAQUE);
	outtextxy(355, 366, str8);
	settextstyle(30, 0, _T("锐字云字库隶书体1.0"), 0, 0, FW_EXTRALIGHT, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	setbkmode(TRANSPARENT);
}
void waiting() {
	settextstyle(150, 0, _T("Zapfino"), 0, 0, FW_BLACK, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	setbkmode(TRANSPARENT);
	settextcolor(BLUE);
	outtextxy(415, 540, str9);
}
void creator() {
	settextstyle(30, 0, _T("宋体"), 0, 0, FW_BOLD, true, false, false);
	settextcolor(BLACK);
	setbkmode(TRANSPARENT);
	outtextxy(600, 670, str7);
}
void exit() {
	settextstyle(30, 0, _T("Arial"), 0, 0, FW_BOLD, true, false, false);
	settextcolor(BLACK);
	setbkmode(TRANSPARENT);
	outtextxy(0, 670, str15);
	settextstyle(60, 0, _T("黑体"), 0, 0, FW_BOLD, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	setbkmode(TRANSPARENT);
	settextcolor(WHITE);
	outtextxy(870, 640, str16);
}
void ifexit() {
	settextstyle(60, 0, _T("黑体"), 0, 0, FW_BOLD, false, false, false);
	LOGFONT f;
	gettextstyle(&f);
	f.lfQuality = ANTIALIASED_QUALITY;
	settextstyle(&f);
	setbkmode(TRANSPARENT);
	settextcolor(RED);
	outtextxy(870, 640, str16);
}
int main()
{
	bool opicture = true;
mark:
	fstream fout("mask.txt", ios::out | ios::trunc);
	char s[20];
	iloadimage();
	initgraph(995, 700);
	int pencilradius = 0;
	bool icircle = 0;
	bool inorout = 0;
	setbkinit();
	setpicinit();
	settextinit();
	exit();
	while (1) {
		MOUSEMSG m = GetMouseMsg();
		if (m.x < 625 && m.x>370 && m.y < 510 && m.y>410) {//图片转换按钮
			if (inorout == true) {
				setbkinit();
				setpicinit();
				settextinit();
				exit();
				inorout = false;
			}
			iftransform();
			if (opicture == false)
				rechoose();
			FlushBatchDraw();
		}
		else if (m.x < 626 && m.x>369 && m.y < 356 && m.y>100) {//插入图片按钮
			if (inorout == true) {
				setbkinit();
				setpicinit();
				settextinit();
				exit();
				inorout = false;
			}
			ifpicture();
			if (opicture == false)
				rechoose();
			FlushBatchDraw();
			if (m.mkLButton) {
				closegraph();
				system("cls");
				cin >> s;
				fstream fin(s, ios::in);
				if (!fin) {
					opicture = false;
					goto mark;
				}
				opicture = true;
				char _s[256];
				sprintf_s(_s, "%s %s", "python reshape_picture.py", s);
				system(_s);
				loadimage(&picture, _T("newpic.jpg"));
				break;
			}
		}
		else if (incircle(m.x, m.y, 650, 225, 15) == true) {//改变涂鸦笔大小
			clickcircle(650, 225, 15);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 15;
				icircle = true;
			}
		}
		else if (incircle(m.x, m.y, 650, 265, 10) == true) {//同上
			clickcircle(650, 265, 10);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 10;
				icircle = true;
			}
		}
		else if (incircle(m.x, m.y, 650, 295, 5) == true) {//同上
			clickcircle(650, 295, 5);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 5;
				icircle = true;
			}
		}
		else if (m.x > 870 && m.x < 995 && m.y>640 && m.y < 700) {
			ifexit();
			FlushBatchDraw();
			if (m.mkLButton)
				return 0;
		}
		else {//其他区域
			inorout = true;
			BeginBatchDraw();
			setbkinit();
			setpicinit();
			settextinit();
			exit();
			if (opicture == false)
				rechoose();
			FlushBatchDraw();
			if (icircle == true) {
				followcircle(m.x, m.y, pencilradius);
				FlushBatchDraw();
			}
			if (m.mkLButton) {
				icircle = false;
			}
		}
	}
	icircle = false;
	mask.clear();
	imask.clear();
	pencilradius = 0;
	inorout = false;
	for (int i = 0; i < 256; i++)
		for (int j = 0; j < 256; j++)
			mmask[i][j] = 0;
	imask.push_back(picture);
	initgraph(995, 700);
	setbkinit();
	setpicinit();
	settextinit();
	setpictureinit();
	creator();
	FlushBatchDraw();
	while (1) {
		MOUSEMSG m = GetMouseMsg();
		switch (m.uMsg) {
		case WM_LBUTTONUP:
			IMAGE curpicture;
			getimage(&curpicture, 369, 100, 256, 256);
			imask.push_back(curpicture);
			mask.clear();
		}
		if (m.x < 625 && m.x>370 && m.y < 510 && m.y>410) {//图片补齐按钮
			if (inorout == true) {
				setbkinit();
				setpicinit();
				settextinit();
				creator();
				topicture();
				inorout = false;
			}
			iftransform();
			FlushBatchDraw();
			if (m.mkLButton) {
				break;
			}
		}
		else if (m.x < 626 && m.x>369 && m.y < 356 && m.y>100) {//涂鸦
			if (inorout == true) {
				setbkinit();
				setpicinit();
				settextinit();
				creator();
				topicture();
				if (m.x < 626 - pencilradius && m.x > 369 + pencilradius && m.y < 356 - pencilradius && m.y > 100 + pencilradius)
					inorout = false;
			}
			else {
				topicture();
				tomask();
			}
			FlushBatchDraw();
			if (icircle == true) {
				followcircle(m.x, m.y, pencilradius);
				FlushBatchDraw();
			}
			if (m.mkLButton) {
				if (icircle == true) {
					ccircle point(m.x, m.y, pencilradius);
					mask.push_back(point);
					getmask(m.x - 369, m.y - 100, pencilradius);
				}
			}
		}
		else if (incircle(m.x, m.y, 650, 225, 15) == true) {//改变涂鸦笔大小
			inorout = true;
			clickcircle(650, 225, 15);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 15;
				icircle = true;
			}
		}
		else if (incircle(m.x, m.y, 650, 265, 10) == true) {//同上
			inorout = true;
			clickcircle(650, 265, 10);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 10;
				icircle = true;
			}
		}
		else if (incircle(m.x, m.y, 650, 295, 5) == true) {//同上
			inorout = true;
			clickcircle(650, 295, 5);
			FlushBatchDraw();
			if (m.mkLButton) {
				pencilradius = 5;
				icircle = true;
			}
		}
		else {//其他区域
			inorout = true;
			BeginBatchDraw();
			setbkinit();
			setpicinit();
			settextinit();
			creator();
			topicture();
			tomask();
			FlushBatchDraw();
			if (icircle == true) {
				followcircle(m.x, m.y, pencilradius);
				FlushBatchDraw();
			}
			if (m.mkLButton) {
				icircle = false;
				pencilradius = 0;
			}
		}
	}
	for (int i = 0; i < 256; i++) {
		for (int j = 0; j < 256; j++)
			fout << mmask[i][j] << ' ';
		fout << endl;
	}
	//fout << s << endl;
	fout.close();
	setbkinit();
	setpicinit();
	settextinit();
	creator();
	topicture();
	waiting();
	FlushBatchDraw();
	system("python make_mask.py");
	system("python test.py");
	loadimage(&finalpicture, "final.jpg");
	while (1) {
		MOUSEMSG m = GetMouseMsg();
		if (m.x > 310 && m.x < 430 && m.y>500 && m.y < 540) {
			ifsave();
			if (m.mkLButton) {
				closegraph();
				system("cls");
				char ss[256];
				cin >> ss;
				strcat(ss + strlen(ss), ".jpg");
				//ss += ".jpg";
				char _ss[100];
				sprintf_s(_ss, "%s %s %s %s", "if exist ", ss, "del", ss);
				system(_ss);
				sprintf_s(_ss, "%s %s", "copy final.jpg", ss);
				system(_ss);
				initgraph(995, 700);
				setfinalinit();
			}
		}
		else if (m.x > 570 && m.x < 650 && m.y>500 && m.y < 540) {
			ifback();
			if (m.mkLButton) {
				opicture = true;
				goto mark;
			}
		}
		else
			setfinalinit();
		FlushBatchDraw();
	}
	system("pause");
	return 0;
}