from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.renderers import render_to_response

import mysql.connector as mysql
import os

db_user = os.environ['MYSQL_USER']
db_pass = os.environ['MYSQL_PASSWORD']
db_name = os.environ['MYSQL_DATABASE']
db_host = os.environ['MYSQL_HOST']

def show_main_page(req):
  return render_to_response('templates/main_page.html', {}, request=req)
  
def show_team_page(req):
  return render_to_response('templates/team.html', {}, request=req)

def show_about_page(req):
  return render_to_response('templates/about.html', {}, request=req)   

''' Route Configurations '''
if __name__ == '__main__':
  config = Configurator()

  config.include('pyramid_jinja2')
  config.add_jinja2_renderer('.html')

  config.add_route('get_welcome', '/')
  config.add_view(show_main_page, route_name='get_welcome')
  
  config.add_route('team_page', '/team')
  config.add_view(show_team_page, route_name='team_page')
  
  config.add_route('about_page', '/about')
  config.add_view(show_about_page, route_name='about_page')

  config.add_static_view(name='/', path='./public', cache_max_age=3600)

  app = config.make_wsgi_app()
  server = make_server('0.0.0.0', 6000, app)
  server.serve_forever()