---
layout: author
permalink: /categories/
title: Categories
image: Categories.jpg
---

<style>
  #archives {
    font-family: Arial, sans-serif;
    padding: 20px;
    max-width: 800px;
    margin: 0 auto;
  }

  .archive-group {
    border: 1px solid #e0e0e0;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 5px;
    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
  }

  .category-head {
    margin: 0;
    padding: 10px;
    background-color: #007bff;
    color: #fff;
    border-radius: 5px;
    text-transform: uppercase;
  }

  .post-list {
    margin: 0;
    padding: 0;
    list-style-type: none;
  }

  .archive-item {
    margin: 10px 0;
  }

  .post-link {
    text-decoration: none;
    color: #EA950B;
    transition: color 0.3s;
  }

  .post-link:hover {
    color: #bf7409;
  }
</style>


<div id="archives">
  {% for category in site.categories %}
    {% capture category_name %}{{ category | first }}{% endcapture %}
    <section class="archive-group">
      <h4 class="category-head" id="{{ category_name | slugize }}">
        {{ category_name }}
      </h4>
      <ul class="post-list">
        {% for post in site.categories[category_name] %}
          <li class="archive-item">
            <a href="{{ site.baseurl }}{{ post.url }}" class="post-link">{{ post.title }}</a>
          </li>
        {% endfor %}
      </ul>
    </section>
  {% endfor %}
</div>
