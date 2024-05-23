from django.urls import path, include
from django.http import HttpResponseRedirect
from . import views
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('mlmodel/', include(('mlmodel.urls', 'mlmodel'), namespace='mlmodel')),
    path('', views.home, name='home'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
