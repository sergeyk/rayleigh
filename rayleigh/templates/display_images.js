var items = $.map(json_data['results'], function(val, i) {
  var id = val['id'];
  var url = val['url'];
  var distance = sprintf('%.2f', val['distance']);

  var a_search_by_image = sprintf('<a href="%s">search by image</a>',
    sprintf('/search_by_image/%s/%s', sic_type, id));
  var img = sprintf('<img src="%s" alt="%.3f" width="%d" height="%d" />',
    url, val['distance'], val['width']/2, val['height']/2);
  var img_link = sprintf('<a href=%s>%s</a>', url, img);
  var caption = [distance, a_search_by_image].join(' | ');
  return sprintf('<div class="result">%s<br />%s</div>', img_link, caption)
});

var time = sprintf('Time elapsed: %.2f ms.<br />', json_data['time_elapsed']*1000);
$('#results').html(time+items.join(''));