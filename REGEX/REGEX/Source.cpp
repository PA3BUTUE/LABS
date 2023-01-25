#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>

using namespace std;

int main()
{
	smatch match;
	const auto reg = regex(R"([a-zA-Z0-9_-]+@[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)");

	ifstream file("page.html");
	string line;
	while (getline(file, line))
	{
		regex_search(line, match, reg);
		if (!match.empty()) cout << match[0] << endl;
	}
}