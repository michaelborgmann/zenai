$(function() {

  $("input,textarea").not("[type=search]").jqBootstrapValidation({

    preventSubmit: true,
    submitError: function($form, event, errors) { },
    submitSuccess: function($form, e) {

      e.preventDefault();

      var submitButton = $('input[type=submit]', $form);

      $.ajax({
        type: 'POST',
        url: $form.prop('action'),
    	  accept: { javascript: 'application/javascript' },
    	  data: $form.serialize(),
    	  beforeSend: function() {
          submitButton.prop('value', 'Sending...');
          submitButton.prop('disabled', 'disabled');
        }
      }).done(function(data) {

        submitButton.prop('value', 'Thanks, I\x27ll come back to you soon.');
        submitButton.prop('disabled', false);
      });
    },

    filter: function() {
      return $(this).is(":visible");
    },
  });
});

$('#name').focus(function() {
  $('#success').html('');
});
