git symbolic-ref HEAD refs/heads/local
git reset
pre-commit uninstall
# edit .gitignore
git reset --soft HEAD~1
git add *
git commit -m "sync"
git push --set-upstream origin local -f

git symbolic-ref HEAD refs/heads/global_pytorch_forecasting
git reset
pre-commit install
# edit .gitignore
git commit -m "some message"
