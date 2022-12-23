##Import   
  <link type="text/css" href="~/static/select2/css/select2.min.css" rel="stylesheet" />
  <script src="~/static/select2/js/select2.min.js"></script>

  <script>
     $(document).on('select2:open', function (e) {
          const selectId = e.target.id;
          $(".select2-search__field[aria-controls='select2-" + selectId + "-results']").each(function                 (key, value) {
              value.focus();
          });
      });
      
      
    $(document).ready(function () {

        $("#selectdropdown1").select2({
            placeholder: "All",
            allowClear: true,
            width: 'resolve'
        });
        
        $("#selectdropdown2").select2({
            placeholder: "",
            allowClear: true,
            width: 'resolve'
        });
    });
  </script>
