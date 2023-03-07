tee app_reachy_face_tracking.service <<EOF
[Unit]
Description=Reachy face tracking service

[Service]
ExecStart=/usr/bin/bash $PWD/launch.bash

[Install]
WantedBy=default.target
EOF

mkdir -p $HOME/.config/systemd/user

mv app_reachy_face_tracking.service $HOME/.config/systemd/user

echo ""
echo "app_reachy_face_tracking.service is now setup."