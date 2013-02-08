var items = $.map(json_data['results'], function(val, i) {
  var id = val['id'];
  var url = val['url'];
  var a_search_by_image = sprintf('<a href="%s">search by this image</a>',
    sprintf('/search_by_image/%s/%s', sic_type, id));
  var img = sprintf('<img src="%s" alt="%.3f" width="%d" height="%d" />',
    url, val['distance'], val['width']/2, val['height']/2);
  var img_link = sprintf('<a href=%s>%s</a>', url, img);
  var links = [a_search_by_image].join(' | ');
  return sprintf('<div class="result">%s<br />%s</div>', img_link, links)
});
$('#images').html(items.join(''));