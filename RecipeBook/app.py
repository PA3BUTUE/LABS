# from pydoc import render_doc
from flask_sqlalchemy import SQLAlchemy

from flask import Flask, render_template, request, redirect
from sqlalchemy import Nullable

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///recipes.db'
db = SQLAlchemy(app)

class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(300), nullable=False)
    description = db.Column(db.Text, nullable=False)
    ingredients = db.Column(db.Text, nullable=False)
    recipe = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(500))


@app.route("/index")
@app.route("/")
def index():
    posts = Post.query.order_by(Post.id.desc()).limit(3).all()
    return render_template("index.html", posts = posts)

@app.route("/recipes")
def recipes():
    posts = Post.query.order_by(Post.id.desc()).all()
    return render_template("recipes.html", posts = posts)

@app.route("/create", methods = ['POST', 'GET'])
def create():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        ingredients = request.form['ingredients']
        recipe = request.form['recipe']
        image_url = request.form['image_url']


        post = Post(title = title, description = description, ingredients = ingredients, recipe = recipe, image_url = image_url)

        try:
            db.session.add(post)
            db.session.commit()
            return redirect("/")
        except:
            return "Ошибка добавления записи в БД"
    else:
        return render_template("create.html")


@app.route('/recipes/<int:post_id>')
def recipe_detail(post_id):
    post = Post.query.get_or_404(post_id)
    return render_template('recipe_detail.html', post=post)

if __name__ == "__main__":
    app.run(debug=True)