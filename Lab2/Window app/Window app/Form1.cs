namespace Window_app
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            int a = int.Parse(textBox1.Text);
            int b = int.Parse(textBox2.Text);

            for (int i = a; i < b; i++)
            {
                bool isSimple = true;
                for (int j = 2; j < i / 2; j++)
                {
                    if (i % j == 0)
                    {
                        isSimple = false;
                        break;
                    }
                }

                if (isSimple) listBox1.Items.Add(i.ToString());
            }
        }
    }
}