use rand::seq::SliceRandom;
use rand::thread_rng;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;

#[allow(dead_code)]
/// chunk the array to `arr` using `size`
fn chunk<T: Clone>(arr: &[T], size: usize) -> Vec<Vec<T>> {
    if size == 0 {
        return Vec::new();
    }

    arr.chunks(size).map(|chunk| chunk.to_vec()).collect()
}

#[allow(dead_code)]
/// gets elements which matches `iarr` elements as index, `narr` elements
fn at<T: Clone>(narr: &[T], iarr: &[usize]) -> Vec<T> {
    iarr.iter()
        .map(|index| {
            if *index > 0 && index <= &narr.len() {
                narr[index - 1].clone()
            } else {
                panic!(
                    "index out of range at int index array, :{}: {}",
                    index,
                    narr.len()
                );
            }
        })
        .collect()
}

#[allow(dead_code)]
/// for compatibility
fn compact(slice: &[Option<i32>]) -> Vec<i32> {
    slice.iter().filter_map(|&x| x).collect()
}

#[allow(dead_code)]
/// concat the array
fn concat<T>(rv: Vec<T>, values: &[T]) -> Vec<T>
where
    T: Clone + Hash + Eq + Ord,
{
    let mut set: HashSet<T> = rv.into_iter().collect();
    set.extend(values.iter().cloned());

    let mut res: Vec<T> = set.into_iter().collect();
    res.sort();

    res
}

#[allow(dead_code)]
fn count_by<T, K>(items: &[T], f: K) -> HashMap<String, usize>
where
    K: Fn(&T) -> String,
{
    let mut counts = HashMap::new();

    for item in items {
        let key = f(item);
        let count = counts.entry(key).or_insert(0);
        *count += 1;
    }

    counts
}

#[allow(dead_code)]
fn difference<T: PartialEq + Clone>(arr1: &[T], arr2: &[T]) -> Vec<T> {
    arr1.iter()
        .filter(|&item| !arr2.contains(item))
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn difference_by<T, K>(arr1: &[T], arr2: &[T], iteratree: K) -> Vec<T>
where
    T: Clone + Eq + Hash,
    K: Fn(&T) -> T,
{
    let set: HashSet<_> = arr2.iter().map(&iteratree).collect();
    arr1.iter()
        .filter(|&i| !set.contains(&iteratree(i)))
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn difference_with<T, F>(arr1: &[T], arr2: &[T], comparator: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    arr1.iter()
        .filter(|&item1| !arr2.iter().any(|item2| comparator(item1, item2)))
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn drop<T: Clone>(arr: &[T], n: usize) -> Vec<T> {
    arr.iter().skip(n).cloned().collect()
}

#[allow(dead_code)]
fn drop_right<T: Clone>(arr: &[T], n: usize) -> Vec<T> {
    if arr.len() > n {
        arr[..arr.len() - n].to_vec()
    } else {
        Vec::new()
    }
}

#[allow(dead_code)]
fn drop_right_while<T, F>(vec: &mut Vec<T>, predicate: F)
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    while let Some(last) = vec.last() {
        if predicate(last) {
            vec.pop();
        } else {
            break;
        }
    }
}

#[allow(dead_code)]
fn drop_while<T, F>(vec: &mut Vec<T>, predicate: F)
where
    T: Clone,
    F: Fn(&T) -> bool,
{
    let mut i = 0;

    // Find the index where the predicate is false
    while i < vec.len() && predicate(&vec[i]) {
        i += 1;
    }

    // Keep only the elements from index `i` onwards
    vec.drain(0..i);
}

#[allow(dead_code)]
fn fill<T>(vec: &mut Vec<T>, value: T)
where
    T: Clone,
{
    vec.iter_mut().for_each(|elem| *elem = value.clone());
}

#[allow(dead_code)]
fn fill_range<T>(vec: &mut Vec<T>, value: T, start: usize, end: usize)
where
    T: Clone,
{
    let len = vec.len();
    if start >= len {
        return; // Start index is out of bounds
    }
    let end = end.min(len);
    for elem in &mut vec[start..end] {
        *elem = value.clone();
    }
}

#[allow(dead_code)]
fn find<T, F>(vec: &[T], predicate: F) -> Option<&T>
where
    T: PartialEq + Debug, // Ensure T can be compared and printed
    F: Fn(&T) -> bool,
{
    vec.iter().find(|&&ref item| predicate(item))
}

#[allow(dead_code)]
fn find_index<T, F>(vec: &[T], predicate: F) -> Option<usize>
where
    T: PartialEq + Debug, // Ensure T can be compared and printed
    F: Fn(&T) -> bool,
{
    vec.iter().position(|item| predicate(item))
}

#[allow(dead_code)]
fn flat_map<T, U, F>(vec: Vec<T>, f: F) -> Vec<U>
where
    T: Clone,
    F: Fn(T) -> Vec<U>,
{
    vec.into_iter().flat_map(f).collect()
}

#[allow(dead_code)]
fn flatten<T>(nested: Vec<T>, depth: usize) -> Vec<T>
where
    T: IntoIterator<Item = T> + Clone,
    T::IntoIter: ExactSizeIterator,
{
    let mut result = Vec::new();
    let mut stack: Vec<(T, usize)> = nested.into_iter().map(|item| (item, depth)).collect();

    while let Some((item, current_depth)) = stack.pop() {
        if current_depth == 0 {
            result.push(item);
        } else {
            let mut iter = item.into_iter();
            if let Some(first) = iter.next() {
                stack.push((first, current_depth - 1));
                stack.extend(iter.map(|e| (e, current_depth - 1)));
            }
        }
    }

    result
}

#[allow(dead_code)]
fn flatten_deep<T>(nested: Vec<T>) -> Vec<T>
where
    T: IntoIterator<Item = T> + Clone,
{
    let mut result = Vec::new();
    for item in nested {
        result.extend(flatten_deep(item.into_iter().collect()));
    }
    result
}

#[allow(dead_code)]
fn for_each_right<T, F>(arr: &[T], mut f: F)
where
    F: FnMut(&T),
{
    for item in arr.iter().rev() {
        f(item);
    }
}

#[allow(dead_code)]
fn group_by<T, K, F>(arr: Vec<T>, key_fn: F) -> HashMap<K, Vec<T>>
where
    K: std::cmp::Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut map = HashMap::new();
    for item in arr {
        let key = key_fn(&item);
        map.entry(key).or_insert_with(Vec::new).push(item);
    }
    map
}

#[allow(dead_code)]
fn head<T>(arr: &[T]) -> Option<&T> {
    arr.first()
}

#[allow(dead_code)]
fn index_of<T: PartialEq>(arr: &[T], value: T) -> Option<usize> {
    arr.iter().position(|x| *x == value)
}

#[allow(dead_code)]
fn initial<T>(arr: &[T]) -> &[T] {
    if arr.is_empty() {
        arr // 배열이 비어 있으면 그대로 반환
    } else {
        &arr[..arr.len() - 1] // 마지막 요소를 제외한 슬라이스 반환
    }
}

#[allow(dead_code)]
fn intersection<T: Eq + Hash + Clone>(a: &HashSet<T>, b: &HashSet<T>) -> HashSet<T> {
    a.intersection(b).cloned().collect()
}

#[allow(dead_code)]
fn intersection_by<T, K, F>(a: &[T], b: &[T], key_fn: F) -> Vec<T>
where
    T: Clone + Eq + Hash,
    K: Eq + Hash + Clone,
    F: Fn(&T) -> K,
{
    let set_a: HashSet<_> = a.iter().map(|x| key_fn(x)).collect();
    let set_b: HashSet<_> = b.iter().map(|x| key_fn(x)).collect();

    let common_keys: HashSet<_> = intersection(&set_a, &set_b);

    a.iter()
        .filter(|x| common_keys.contains(&key_fn(x)))
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn intersection_with<T, F>(a: &[T], b: &[T], cmp_fn: F) -> Vec<T>
where
    T: Clone + Eq + Hash,
    F: Fn(&T, &T) -> bool,
{
    // 기준 함수를 사용하여 요소를 비교하기 위해 HashSet을 사용
    let set_b: HashSet<&T> = b.iter().collect();

    a.iter()
        .filter(|&x| set_b.iter().any(|y| cmp_fn(x, y)))
        .cloned()
        .collect()
}

#[allow(dead_code)]
/// Returns `true` if `a` is a subset of `b`, otherwise `false`.
fn is_subset<T>(a: &HashSet<T>, b: &HashSet<T>) -> bool
where
    T: Eq + Hash,
{
    a.iter().all(|item| b.contains(item))
}

#[allow(dead_code)]
/// Groups the elements of `items` by the key generated by `key_fn`.
/// Returns a HashMap where the keys are the result of applying `key_fn` to each item,
/// and the values are vectors of items that share the same key.
fn key_by<T, K, F>(items: &[T], key_fn: F) -> HashMap<K, Vec<T>>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut map = HashMap::new();

    for item in items {
        let key = key_fn(item);
        map.entry(key).or_insert_with(Vec::new).push(item.clone());
    }

    map
}

#[allow(dead_code)]
fn last<T>(items: &[T]) -> Option<&T> {
    if items.is_empty() {
        None
    } else {
        Some(&items[items.len() - 1])
    }
}

#[allow(dead_code)]
fn max<T>(items: &[T]) -> Option<&T>
where
    T: Ord,
{
    items.iter().max()
}

#[allow(dead_code)]
/// Finds the maximum element in a slice using a comparison function.
/// Returns `Some(&T)` if the slice is not empty, `None` otherwise.
fn max_by<T, F>(items: &[T], compare: F) -> Option<&T>
where
    T: Ord,
    F: Fn(&T, &T) -> Ordering,
{
    items.iter().max_by(|a, b| compare(a, b))
}

#[allow(dead_code)]
/// Finds the minimum element in a slice.
/// Returns `Some(&T)` if the slice is not empty, `None` otherwise.
fn min<T>(items: &[T]) -> Option<&T>
where
    T: Ord,
{
    items.iter().min()
}

#[allow(dead_code)]
/// Finds the minimum element in a slice using a comparison function.
/// Returns `Some(&T)` if the slice is not empty, `None` otherwise.
fn min_by<T, F>(items: &[T], compare: F) -> Option<&T>
where
    T: Ord,
    F: Fn(&T, &T) -> Ordering,
{
    items.iter().min_by(|a, b| compare(a, b))
}

#[allow(dead_code)]
fn order_by<T, F>(items: &mut Vec<T>, criteria: Vec<F>, orders: Vec<&str>)
where
    F: Fn(&T) -> Ordering,
{
    items.sort_by(|a, b| {
        for (criterion, order) in criteria.iter().zip(orders.iter()) {
            let cmp = criterion(a).cmp(&criterion(b));
            if cmp != Ordering::Equal {
                return if *order == "asc" { cmp } else { cmp.reverse() };
            }
        }
        std::cmp::Ordering::Equal
    });
}

#[allow(dead_code)]
fn partition<T, F>(items: &[T], predicate: F) -> (Vec<T>, Vec<T>)
where
    F: Fn(&T) -> bool,
    T: Clone,
{
    let mut truthy = Vec::new();
    let mut falsy = Vec::new();

    for item in items {
        if predicate(item) {
            truthy.push(item.clone());
        } else {
            falsy.push(item.clone());
        }
    }

    (truthy, falsy)
}

#[allow(dead_code)]
fn pull_at<T>(arr: &mut Vec<T>, indexes: &[usize]) -> Vec<T>
where
    T: Clone,
{
    let mut removed = Vec::new();

    // 인덱스를 정렬하여 안전하게 제거
    let mut sorted_indexes = indexes.to_vec();
    sorted_indexes.sort_unstable();
    sorted_indexes.reverse(); // 큰 인덱스부터 제거

    for &index in &sorted_indexes {
        if index < arr.len() {
            removed.push(arr.remove(index));
        }
    }

    removed
}

#[allow(dead_code)]
fn sample<T>(arr: &[T]) -> Option<&T> {
    let mut rng = thread_rng();
    arr.choose(&mut rng) // 무작위로 하나의 요소 선택
}

#[allow(dead_code)]
fn sample_size<T>(arr: &[T], n: usize) -> Vec<&T>
where
    T: Clone,
{
    let mut rng = thread_rng();
    let mut chosen = vec![];

    let sample_count = n.min(arr.len()); // n이 배열 길이를 초과하지 않도록 조정
    let indices: Vec<usize> = (0..arr.len()).collect();

    let random_indices = indices.choose_multiple(&mut rng, sample_count);

    for &index in random_indices {
        chosen.push(&arr[index]);
    }

    chosen
}

#[allow(dead_code)]
fn shuffle<T>(arr: &mut [T]) {
    let mut rng = thread_rng();
    arr.shuffle(&mut rng); // 배열을 무작위로 섞음
}

#[allow(dead_code)]
fn size<T>(arr: &[T]) -> usize {
    arr.len() // 배열의 길이를 반환
}

#[allow(dead_code)]
fn sort_by<T, F>(arr: &mut [T], compare_fn: F)
where
    F: Fn(&T, &T) -> Ordering,
{
    arr.sort_by(compare_fn); // 비교 함수를 사용하여 정렬
}

#[allow(dead_code)]
fn tail<T>(arr: &[T]) -> &[T] {
    if arr.is_empty() {
        &[] // 빈 배열인 경우 빈 슬라이스 반환
    } else {
        &arr[1..] // 첫 번째 요소를 제외한 슬라이스 반환
    }
}

#[allow(dead_code)]
fn take<T>(arr: &[T], count: usize) -> &[T] {
    if count > arr.len() {
        arr // 지정된 수가 배열 길이보다 크면 전체 배열 반환
    } else {
        &arr[..count] // 처음 count 개수만큼의 슬라이스 반환
    }
}

#[allow(dead_code)]
fn take_right<T>(arr: &[T], count: usize) -> &[T] {
    if count >= arr.len() {
        arr // 지정된 수가 배열 길이보다 크면 전체 배열 반환
    } else {
        &arr[arr.len() - count..] // 끝에서 count 개수만큼의 슬라이스 반환
    }
}

#[allow(dead_code)]
fn take_right_while<T, F>(arr: &[T], predicate: F) -> Vec<&T>
where
    F: Fn(&T) -> bool,
{
    let mut result = Vec::new();
    for item in arr.iter().rev() {
        if predicate(item) {
            result.push(item);
        } else {
            break; // 조건을 만족하지 않는 첫 번째 요소에서 종료
        }
    }
    result.reverse(); // 원래 순서로 반환
    result
}

#[allow(dead_code)]
fn take_while<T, F>(arr: &[T], predicate: F) -> Vec<&T>
where
    F: Fn(&T) -> bool,
{
    let mut result = Vec::new();
    for item in arr.iter() {
        if predicate(item) {
            result.push(item);
        } else {
            break; // 조건을 만족하지 않는 첫 번째 요소에서 종료
        }
    }
    result
}

#[allow(dead_code)]
fn to_filled<T: Clone>(length: usize, value: T) -> Vec<T> {
    vec![value; length] // 주어진 값으로 채워진 벡터 생성
}

#[allow(dead_code)]
fn union<T: Eq + Hash + Clone>(arr1: &[T], arr2: &[T]) -> Vec<T> {
    let mut set = HashSet::new();
    set.extend(arr1.iter().cloned());
    set.extend(arr2.iter().cloned());
    set.into_iter().collect()
}

#[allow(dead_code)]
/// Computes the union of two slices using a key function to determine uniqueness.
/// Returns a vector containing the unique elements from both slices.
fn union_by<T, K, F>(a: &[T], b: &[T], key_fn: F) -> Vec<T>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for item in a.iter().chain(b.iter()) {
        let key = key_fn(item);
        if seen.insert(key) {
            result.push(item.clone());
        }
    }

    result
}

#[allow(dead_code)]
/// Computes the union of two slices using a merge function to handle duplicates.
/// Returns a vector containing the merged elements from both slices.
fn union_with<T, K, F>(a: &[T], b: &[T], key_fn: fn(&T) -> K, merge_fn: F) -> Vec<T>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(T, T) -> T,
{
    let mut map: HashMap<K, T> = HashMap::new();

    for item in a.iter().chain(b.iter()) {
        let key = key_fn(item);
        if let Some(existing_item) = map.get_mut(&key) {
            *existing_item = merge_fn(existing_item.clone(), item.clone());
        } else {
            map.insert(key, item.clone());
        }
    }

    map.into_values().collect()
}

#[allow(dead_code)]
fn uniq<T: Eq + Hash + Clone>(items: &[T]) -> Vec<T> {
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for item in items {
        if seen.insert(item.clone()) {
            result.push(item.clone());
        }
    }

    result
}

#[allow(dead_code)]
/// Removes duplicate elements from a slice based on a key function.
/// Returns a vector of unique elements where uniqueness is determined by the key function.
fn uniq_by<T, K, F>(items: &[T], key_fn: F) -> Vec<T>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let mut seen = HashSet::new();
    let mut result = Vec::new();

    for item in items {
        let key = key_fn(item);
        if seen.insert(key) {
            result.push(item.clone());
        }
    }

    result
}

#[allow(dead_code)]
/// Removes duplicate elements from a slice based on a comparison function.
/// Returns a vector of unique elements where uniqueness is determined by the comparison function.
fn uniq_with<T, F>(items: &[T], eq_fn: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    let mut result = Vec::new();

    for item in items {
        if !result.iter().any(|existing| eq_fn(existing, item)) {
            result.push(item.clone());
        }
    }

    result
}

#[allow(dead_code)]
fn unzip<T, U>(pairs: Vec<(T, U)>) -> (Vec<T>, Vec<U>) {
    pairs.into_iter().unzip()
}

#[allow(dead_code)]
/// Returns a new vector that excludes the specified values.
fn without<T: Eq + Hash + Clone>(items: &[T], exclude: &[T]) -> Vec<T> {
    let exclude_set: HashSet<_> = exclude.iter().collect();
    items
        .iter()
        .filter(|item| !exclude_set.contains(item))
        .cloned()
        .collect()
}

#[allow(dead_code)]
fn xor<T: Eq + Hash + Clone>(a: &[T], b: &[T]) -> Vec<T> {
    let set_a: HashSet<_> = a.iter().cloned().collect();
    let set_b: HashSet<_> = b.iter().cloned().collect();

    let in_a_not_in_b = set_a.difference(&set_b).cloned().collect::<Vec<_>>();
    let in_b_not_in_a = set_b.difference(&set_a).cloned().collect::<Vec<_>>();

    in_a_not_in_b
        .into_iter()
        .chain(in_b_not_in_a.into_iter())
        .collect()
}

#[allow(dead_code)]
fn xor_by<T, K, F>(a: &[T], b: &[T], key_fn: F) -> Vec<T>
where
    T: Clone,
    K: Eq + Hash,
    F: Fn(&T) -> K,
{
    let set_a: HashSet<_> = a.iter().map(&key_fn).collect();
    let set_b: HashSet<_> = b.iter().map(&key_fn).collect();

    let in_a_not_in_b = a
        .iter()
        .filter(|item| !set_b.contains(&key_fn(item)))
        .cloned()
        .collect::<Vec<_>>();
    let in_b_not_in_a = b
        .iter()
        .filter(|item| !set_a.contains(&key_fn(item)))
        .cloned()
        .collect::<Vec<_>>();

    in_a_not_in_b
        .into_iter()
        .chain(in_b_not_in_a.into_iter())
        .collect()
}

#[allow(dead_code)]
fn xor_with<T, F>(a: &[T], b: &[T], cmp_fn: F) -> Vec<T>
where
    T: Clone,
    F: Fn(&T, &T) -> bool,
{
    let mut result = Vec::new();

    for item in a {
        if !b.iter().any(|x| cmp_fn(item, x)) && !result.iter().any(|x| cmp_fn(item, x)) {
            result.push(item.clone());
        }
    }

    for item in b {
        if !a.iter().any(|x| cmp_fn(item, x)) && !result.iter().any(|x| cmp_fn(item, x)) {
            result.push(item.clone());
        }
    }

    result
}

#[allow(dead_code)]
fn camel_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = false;

    for (i, word) in s.split_whitespace().enumerate() {
        if i == 0 {
            result.push_str(&word.to_lowercase());
        } else {
            let mut chars = word.chars();
            if let Some(first_char) = chars.next() {
                if capitalize_next {
                    result.push_str(&first_char.to_uppercase().to_string());
                } else {
                    result.push(first_char);
                }
                result.push_str(&chars.as_str().to_lowercase());
            }
        }
        capitalize_next = true;
    }

    result
}

#[allow(dead_code)]
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    if let Some(first_char) = chars.next() {
        // 대문자로 변환한 첫 글자와 나머지 문자열을 결합
        format!("{}{}", first_char.to_uppercase(), chars.as_str())
    } else {
        // 빈 문자열인 경우 그대로 반환
        s.to_string()
    }
}

#[allow(dead_code)]
fn kebab_case(s: &str) -> String {
    let mut result = String::new();
    let mut was_last_char_whitespace = false;

    for c in s.chars() {
        if c.is_whitespace() {
            if !was_last_char_whitespace && !result.is_empty() {
                result.push('-');
            }
            was_last_char_whitespace = true;
        } else {
            result.push(c.to_lowercase().next().unwrap());
            was_last_char_whitespace = false;
        }
    }

    result
}

#[allow(dead_code)]
fn repeat(s: &str, times: usize) -> String {
    s.repeat(times)
}
