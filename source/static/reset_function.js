function reset_all(turn_on_bool) {
    $.post("/reset_button_receiver", turn_on_bool);
    location.reload();
}