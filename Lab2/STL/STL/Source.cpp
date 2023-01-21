//Напишите программу, которая считывает строку и распечатывает её в обратной последовательности.Используйте класс vector

#include <iostream>
#include <vector>

using namespace std;

int main()
{
	vector<char> arr;

	char a;
	a = cin.get();
	while (a != '\n')
	{
		arr.push_back(a);
		a = cin.get();
	}

	for (int i = arr.size() - 1; i >= 0; i--)
	{
		cout << arr[i];
	}
}
