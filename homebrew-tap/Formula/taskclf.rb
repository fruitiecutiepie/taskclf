class Taskclf < Formula
  include Language::Python::Virtualenv

  desc "Local-first task classifier from computer activity signals"
  homepage "https://github.com/fruitiecutiepie/taskclf"
  url "https://files.pythonhosted.org/packages/source/t/taskclf/taskclf-0.1.0.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256"
  license "MIT"

  depends_on "python@3.14"

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"taskclf", "--help"
  end
end
