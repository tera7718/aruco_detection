cmake_minimum_required(VERSION 3.8)
project(item_tools_interfaces)          # ← “__” は不要。パッケージ名どおり

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## ───── ① 生成ツールと依存メッセージを見つける ─────
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)   # ★ 追加
find_package(geometry_msgs REQUIRED)               # ★ 追加

## ───── インタフェースを生成 ─────
rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/GetItemPoses.srv"
  DEPENDENCIES geometry_msgs
)

## ───── テスト・パッケージ宣言 ─────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_dependencies(rosidl_default_runtime)  # ★ RUNTIME 依存を輸出
ament_package()
